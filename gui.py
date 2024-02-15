import streamlit as st
import os
import signal
import math
import warnings
warnings.simplefilter(action='ignore', category=DeprecationWarning)
import pandas as pd
import json
from pandas.io.json._normalize import json_normalize  # for pandas version < 1.0.0 use pd.io.json.json_normalize
import plotly.figure_factory as ff
import time
import subprocess
import psutil

from dotenv import load_dotenv
if not load_dotenv("../.env"):
    load_dotenv(".env")

if "info" not in st.session_state:
    st.session_state.info = None

if "running" not in st.session_state:
    st.session_state.running = False

if "proc" not in st.session_state:
    st.session_state.proc = None

if "model" not in st.session_state:
    st.session_state.model = "gpt-4-turbo"

if "rate" not in st.session_state:
    st.session_state.rate = 20

if "shape_profile" not in st.session_state:
    st.session_state.shape_profile = "context"

AZURE_API_KEY = os.environ["OPENAI_API_KEY"]
AZURE_API_URL = os.environ['OPENAI_API_URL']

if AZURE_API_KEY == "" or AZURE_API_URL == "":
    st.error("Please set the OPENAI_API_KEY and OPENAI_API_URL environment variables")


# Example - Run a single batch of the following two configuration for 120 seconds each, making sure to warm up the PTU-M endpoint prior to each run:

# context_tokens=500, max_tokens=100, rate=20
# context_tokens=3500, max_tokens=300, rate=7.5
# python -m benchmark.contrib.batch_runner https://gbb-ea-openai-swedencentral-01.openai.azure.com/ 
#     --deployment gpt-4-1106-ptu 
#     --token-rate-workload-list 500-100-20,3500-300-7.5 
#     --duration 130 
#     --aggregation-window 120 
#     --log-save-dir logs/ 
#     --start-ptum-runs-at-full-utilization true
#
# 
#################################################################################
# App elements

st.set_page_config(layout="wide")
st.title("Azure OpenAI PTU Benchmark")

def format_num(x):
    if math.isnan(x):
        return int(0.0)
    else:
        return int(x)

# Use the get method since the keys won't be in session_state
# on the first script run

with st.expander("Settings"):
    st.caption("To run the benchmark, make sure to set the OPENAI_API_KEY and OPENAI_API_URL environment variables")

    st.session_state.model = st.selectbox("Select a model", [  "gpt-4-turbo", "gpt-4","gpt-35-turbo"])
    st.session_state.rate = st.slider("Rate", 1, 300, 5, 1)
    st.session_state.shape_profile = st.selectbox("Shape", ["context", "balanced", "generation"])

    if st.session_state.shape_profile == "balanced":
        context_tokens = 500
        max_tokens = 500
    elif st.session_state.shape_profile == "context":
        context_tokens = 2000
        max_tokens = 200
    elif st.session_state.shape_profile == "generation":
        context_tokens = 500
        max_tokens = 1000

    load_max = max(st.session_state.rate - 7,1) * (context_tokens + max_tokens)
    load_min = max(st.session_state.rate - 20,1) * (context_tokens + max_tokens)
    formatted_load_min = f"{load_min//1000}K"
    formatted_load_max = f"{load_max//1000}K"
    st.caption(f"Based on your settings, the benchmark generated load will range from ```{formatted_load_min}``` to ```{formatted_load_max}```\n\n---")

if st.session_state.get('run'):
    st.session_state.running = True
    
    my_env = os.environ.copy()
    my_env["OPENAI_API_KEY"] = AZURE_API_KEY
    
    output_folder = "test_results"
    command = f"python -m benchmark.bench load --deployment {st.session_state.model} --rate {st.session_state.rate} --shape-profile {st.session_state.shape_profile} --clients 80 --output-format jsonl --log-save-dir {output_folder} {AZURE_API_URL}"
    # print (command)

    st.session_state.proc = subprocess.Popen(command.split(), env=my_env)
    time.sleep(3)
    
    poll = st.session_state.proc.poll()
    if poll is None:
        st.session_state.info = st.info(f"Running benchmark as {st.session_state.proc.pid}")
    else:
        st.session_state.info = st.error(f"Error running benchmark: {poll}")
    
if st.session_state.get('stop'):
    st.session_state.running = False

    parent_pid = st.session_state.proc.pid   # my example
    parent = psutil.Process(parent_pid)
    for child in parent.children(recursive=True):  # or parent.children() for recursive=False
        child.kill()
    parent.kill()

col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
with col2:
    st.button('RUN', key='run', disabled=st.session_state.running)
with col3:
    st.button('STOP', key='stop', type="primary", disabled=not st.session_state.running)


st.title("Benchmark Results:")
# create refresh button
if st.button("Refresh"):
    pass

if st.session_state.running:
    st.session_state.info = st.warning(f"Running benchmark as {st.session_state.proc.pid} (params: {st.session_state.model}, {st.session_state.rate}, {st.session_state.shape_profile})")
else:
    st.session_state.info = None

TESTS_RESULTS_FOLDER = "test_results"
TESTS_RESULTS_FILE = "o.jsonl"
# check if folder TESTS_RESULTS_FOLDER exists and if not, create it
if not os.path.exists(TESTS_RESULTS_FOLDER):
    os.makedirs(TESTS_RESULTS_FOLDER)

# list all CSV files in TESTS_RESULTS_FOLDER
files = [f for f in os.listdir(TESTS_RESULTS_FOLDER) if f.endswith('.log')]

selected_file = None
if len(files) > 0:
    # create a dropdown list of files
    selected_file = st.selectbox('Select a file', files)

if selected_file:
    # st.session_state.info = st.info(f"Selected file: {selected_file}")
    TESTS_RESULTS_FILE = selected_file

    # selected_file = os.path.join(TESTS_RESULTS_FOLDER, selected_file)
    data = []
    with open(os.path.join(TESTS_RESULTS_FOLDER,TESTS_RESULTS_FILE), 'r') as f:
        for line in f:
            if line.strip().startswith('{'):
                data.append(json.loads(line))



    # get file timestamp
    file_timestamp = os.path.getmtime(os.path.join(TESTS_RESULTS_FOLDER,TESTS_RESULTS_FILE))

    # format file timestamp to human readable format
    file_timestamp = pd.to_datetime(file_timestamp, unit='s').strftime('%Y-%m-%d %H:%M:%S')

    # get current timestamp in human readable format
    current_timestamp = pd.to_datetime('now').strftime('%Y-%m-%d %H:%M:%S')


    st.caption(f"Resutls from **{TESTS_RESULTS_FILE}** as of {current_timestamp}, file timestamp: {file_timestamp}")

    # Flatten the nested JSON objects
    if (len(data) > 0):
        df = pd.concat([json_normalize(d) for d in data], ignore_index=True)

        # drop rows where 'rpm' is 'n/a'
        df = df[df['rpm'] != 'n/a']
        df = df.replace('n/a', None)

        # get number of rows and columns
        # rows, cols = df.shape
        # print(f"Rows: {rows}, Columns: {cols}")

        # Convert the timestamp column to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # calculate failure_clean as failures - failures from previous row
        df['failure_clean'] = df['failures'] - df['failures'].shift(1)    
        df['throttled_clean'] = df['throttled'] - df['throttled'].shift(1)   

        metric_cols = ['tpm.total','rpm', 'processing', 'completed',
        'failures', 'throttled', 'requests', 'tpm.context', 'tpm.gen',
        'e2e.avg', 'e2e.95th', 'ttft.avg', 'ttft.95th', 'tbt.avg',
        'tbt.95th', 'util.avg', 'util.95th', 'failure_clean',
        'throttled_clean']

        # Convert the 'x' column to numeric type
        for col in metric_cols:
            df[col] = pd.to_numeric(df[col])
        # df['tpm.total'] = pd.to_numeric(df['tpm.total'])
        # df['failures'] = pd.to_numeric(df['failures'])
        # df['rpm'] = pd.to_numeric(df['rpm'])
        # df['requests'] = pd.to_numeric(df['requests'])
        # df['e2e.avg'] = pd.to_numeric(df['e2e.avg'])
        # df['ttft.avg'] = pd.to_numeric(df['ttft.avg'])

        

        
        # dashboard with key metrics
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

        with col1:
            container = st.container(border=True)
            container.caption("Average TPM")
            #f"{load_min//1000}K"
            container.title(f"{format_num(round(df['tpm.total'].mean(), 0))//1000}K")

            container = st.container(border=True)
            container.caption("Max RPM")
            container.title(format_num(df['rpm'].max()))

        with col2:
            container = st.container(border=True)
            container.caption("Max TPM")
            container.title(f"{format_num(round(df['tpm.total'].max(), 0))//1000}K")

            container = st.container(border=True)
            container.caption("Total requests")
            container.title(format_num(df['requests'].max()))

        with col3:
            container = st.container(border=True)
            container.caption("Total Failures")
            container.title(format_num(df['failure_clean'].sum()))

            container = st.container(border=True)
            container.caption("Avg E2E")
            container.title(round(df['e2e.avg'].mean(), 2))

        with col4:
            container = st.container(border=True)
            container.caption("Total Throttling Events")
            container.title(format_num(df['throttled_clean'].sum()))

            container = st.container(border=True)
            container.caption("Avg TTFT")
            container.title(round(df['ttft.avg'].mean(), 2))
            

        import plotly.express as px
        # df = px.data.gapminder().query("continent == 'Oceania'")

        # create selectbox for y axis
        y_axis = st.multiselect('Select Y axis', metric_cols, ["tpm.total"])
        # y_axis

        colors = {"tpm.total":'green',"failures":"red"}
        fig = px.line(df, x='timestamp', y=y_axis, markers=True, color_discrete_map=colors)
        # fig.show()

        # Plot!
        container = st.container(border=True)
        container.plotly_chart(fig, use_container_width=True)

        fig2 = px.line(df, x='timestamp', y=['failure_clean'], markers=True, color_discrete_map=colors)
        # fig.show()

        # Plot!
        container = st.container(border=True)
        container.plotly_chart(fig2, use_container_width=True)

        st.caption("Detailed data")
        st.write(df)
    else:
        st.warning("There is no data captured in the file!")

else:
    st.warning("No benchmark results found - please run test first!")