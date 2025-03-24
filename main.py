import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from datetime import datetime, timedelta
import calendar

# Set page configuration
st.set_page_config(
    page_title="Rimon Recruiting Dashboard",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e6f3ff;
        border-bottom: 3px solid #4e91dc;
    }
    h1, h2, h3 {
        padding-top: 0.8rem;
    }
    .metric-card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 10px;
    }
    .metric-card h3 {
        margin-bottom: 5px;
        font-size: 1.5rem;
    }
    .metric-card p {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 10px 0;
        color: #2c3e50;
    }
    .small-card {
        font-size: 1.5rem !important;
    }
    .info-box {
        background-color: #e1f1ff;
        border-left: 5px solid #4e91dc;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .stat-trend-positive {
        color: #28a745;
        font-size: 0.9rem;
    }
    .stat-trend-negative {
        color: #dc3545;
        font-size: 0.9rem;
    }
    .divider {
        height: 1px;
        background-color: #e0e0e0;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# App title and header
st.title("Rimon Recruiting Dashboard")
st.markdown("### Comprehensive analytics for legal recruitment performance")

# Function to load and prepare data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Combined_Recruiting_Data.csv', low_memory=False)
        
        # Clean up column names (remove trailing/leading spaces)
        df.columns = df.columns.str.strip()
        
        # Make sure all relevant columns are strings before applying string operations
        if 'Recruit Status' in df.columns:
            df['Recruit Status'] = df['Recruit Status'].astype(str)
            # Now extract the status count and clean status
            df['Status_Count'] = df['Recruit Status'].str.extract(r'\(\ ?(\d+)\ ?\)')
            # Convert to numeric, handling errors
            df['Status_Count'] = pd.to_numeric(df['Status_Count'], errors='ignore')
            
            # Also clean up the status names by removing the counts
            df['Status_Clean'] = df['Recruit Status'].str.extract(r'(.*?)\s*\(', expand=False)
            df['Status_Clean'] = df['Status_Clean'].str.strip()
        
        # Clean up and convert date/time fields
        date_fields = ['Last Activity Time', 'Modified Time', 'Created Time', 'Expected Start Date', 'Closed Date']
        for field in date_fields:
            if field in df.columns:
                df[field + '_Date'] = pd.to_datetime(df[field], errors='coerce')
        
        # Extract numeric values from monetary fields
        money_fields = ['Estimated Book (Projected)', 'Estimated Book (Conservative)', 'Estimated Book (Low)', 'Estimated Book (High)']
        for field in money_fields:
            if field in df.columns:
                # Convert to string first to avoid errors
                df[field] = df[field].astype(str)
                df[field + '_Value'] = df[field].str.extract(r'\$\s*([\d,]+(?:\.\d+)?)')
                # Handle comma removal safely
                if field + '_Value' in df.columns and df[field + '_Value'].notna().any():
                    df[field + '_Value'] = df[field + '_Value'].str.replace(',', '')
                # Convert to float with safe error handling
                df[field + '_Value'] = pd.to_numeric(df[field + '_Value'], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Load the data
df = load_data()

# Check if data loaded successfully
if df.empty:
    st.error("Failed to load data. Please check that the CSV file is in the correct location.")
    st.stop()

# Sidebar for filters
st.sidebar.header("Filters")

# Date range filter - based on 'Last Activity Time' if available
if 'Last Activity Time_Date' in df.columns and not df['Last Activity Time_Date'].isna().all():
    min_date = df['Last Activity Time_Date'].min().date()
    max_date = df['Last Activity Time_Date'].max().date()
    
    # Add quick filters for time periods
    time_filter = st.sidebar.radio(
        "Time Period",
        ["Custom Range", "Last 30 Days", "Last 90 Days", "Last 6 Months", "Last Year", "All Time"]
    )
    
    if time_filter == "Custom Range":
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            mask = (df['Last Activity Time_Date'].dt.date >= start_date) & (df['Last Activity Time_Date'].dt.date <= end_date)
            filtered_df = df[mask]
        else:
            filtered_df = df
    else:
        # Calculate the date range based on the selected option
        end_date = max_date
        if time_filter == "Last 30 Days":
            start_date = end_date - timedelta(days=30)
        elif time_filter == "Last 90 Days":
            start_date = end_date - timedelta(days=90)
        elif time_filter == "Last 6 Months":
            start_date = end_date - timedelta(days=180)
        elif time_filter == "Last Year":
            start_date = end_date - timedelta(days=365)
        else:  # All Time
            start_date = min_date
        
        mask = (df['Last Activity Time_Date'].dt.date >= start_date) & (df['Last Activity Time_Date'].dt.date <= end_date)
        filtered_df = df[mask]
else:
    filtered_df = df

# Add filters for recruit status
if 'Status_Clean' in filtered_df.columns and not filtered_df['Status_Clean'].isna().all():
    status_options = filtered_df['Status_Clean'].dropna().unique().tolist()
    default_status = [s for s in status_options if 'On Hold' not in s]  # Default to all except On Hold
    
    selected_statuses = st.sidebar.multiselect(
        "Recruit Status",
        options=status_options,
        default=default_status
    )
    
    if selected_statuses:
        filtered_df = filtered_df[filtered_df['Status_Clean'].isin(selected_statuses)]

# Practice Group filter
if 'Practice Group / Sector' in filtered_df.columns and not filtered_df['Practice Group / Sector'].isna().all():
    practice_groups = filtered_df['Practice Group / Sector'].dropna().unique().tolist()
    selected_practice_groups = st.sidebar.multiselect(
        "Practice Group / Sector",
        options=practice_groups,
        default=[]
    )
    
    if selected_practice_groups:
        filtered_df = filtered_df[filtered_df['Practice Group / Sector'].isin(selected_practice_groups)]

# Recruit Source filter
if 'Recruit Source' in filtered_df.columns and not filtered_df['Recruit Source'].isna().all():
    sources = filtered_df['Recruit Source'].dropna().unique().tolist()
    selected_sources = st.sidebar.multiselect(
        "Recruit Source",
        options=sources,
        default=[]
    )
    
    if selected_sources:
        filtered_df = filtered_df[filtered_df['Recruit Source'].isin(selected_sources)]

# Book value range filter
if 'Estimated Book (Projected)_Value' in filtered_df.columns and not filtered_df['Estimated Book (Projected)_Value'].isna().all():
    min_book = float(filtered_df['Estimated Book (Projected)_Value'].min())
    max_book = float(filtered_df['Estimated Book (Projected)_Value'].max())
    
    book_range = st.sidebar.slider(
        "Estimated Book Value Range ($)",
        min_value=min_book,
        max_value=max_book,
        value=(min_book, max_book),
        step=10000.0
    )
    
    filtered_df = filtered_df[
        (filtered_df['Estimated Book (Projected)_Value'] >= book_range[0]) & 
        (filtered_df['Estimated Book (Projected)_Value'] <= book_range[1])
    ]

# City filter if available
if 'City' in filtered_df.columns and not filtered_df['City'].isna().all():
    # Convert to string first to avoid errors
    filtered_df['City'] = filtered_df['City'].astype(str)
    cities = filtered_df['City'].dropna().unique().tolist()
    if len(cities) > 1:  # Only show if there are multiple cities
        selected_cities = st.sidebar.multiselect(
            "City",
            options=cities,
            default=[]
        )
        
        if selected_cities:
            filtered_df = filtered_df[filtered_df['City'].isin(selected_cities)]

# Referral source filter
if 'Recruit Referral Details' in filtered_df.columns and not filtered_df['Recruit Referral Details'].isna().all():
    # Convert to string first to avoid errors
    filtered_df['Recruit Referral Details'] = filtered_df['Recruit Referral Details'].astype(str)
    # Use a text input for searching referral sources
    referral_search = st.sidebar.text_input("Search Referral Source:")
    if referral_search:
        filtered_df = filtered_df[filtered_df['Recruit Referral Details'].str.contains(referral_search, case=False, na=False)]

# Display the number of records after filtering
st.sidebar.markdown(f"**{len(filtered_df)} records** match current filters")

# Download filtered data as CSV
if st.sidebar.button("Download Filtered Data"):
    csv = filtered_df.to_csv(index=False)
    st.sidebar.download_button(
        label="Click to Download",
        data=csv,
        file_name="filtered_recruiting_data.csv",
        mime="text/csv",
    )

# Tabs for different sections
tabs = st.tabs(["Overview", "Recruits by Source", "Recruits by Status", "Referral Analysis", 
                "Projected Business", "Timeline Analysis", "Geographic Distribution", "Conversion Funnel"])

with tabs[0]:
    st.header("Key Metrics")
    
    # Create top-level metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Total number of recruits
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Total Recruits</h3>
            <p>{}</p>
        </div>
        """.format(len(filtered_df)), unsafe_allow_html=True)
    
    # Active recruits (not on hold)
    if 'Recruit Status' in filtered_df.columns:
        filtered_df['Recruit Status'] = filtered_df['Recruit Status'].astype(str)
        active_recruits = filtered_df[filtered_df['Recruit Status'].notna() & ~filtered_df['Recruit Status'].str.contains('Z. On Hold', na=False)].shape[0]
    else:
        active_recruits = 0
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>Active Recruits</h3>
            <p>{}</p>
        </div>
        """.format(active_recruits), unsafe_allow_html=True)
    
    # On Hold
    if 'Recruit Status' in filtered_df.columns:
        filtered_df['Recruit Status'] = filtered_df['Recruit Status'].astype(str)
        on_hold = filtered_df[filtered_df['Recruit Status'].str.contains('Z. On Hold', na=False)].shape[0]
    else:
        on_hold = 0
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>On Hold</h3>
            <p>{}</p>
        </div>
        """.format(on_hold), unsafe_allow_html=True)
    
    # Estimated Book Value (if available)
    if 'Estimated Book (Projected)_Value' in filtered_df.columns and not filtered_df['Estimated Book (Projected)_Value'].isna().all():
        total_projected = filtered_df['Estimated Book (Projected)_Value'].sum()
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3>Projected Book Value</h3>
                <p>${:,.0f}</p>
            </div>
            """.format(total_projected), unsafe_allow_html=True)
    
    # Create secondary metrics
    st.subheader("Recruitment Pipeline")
    
    # Create a row of smaller metric cards for each pipeline stage
    cols = st.columns(8)
    
    if 'Status_Clean' in filtered_df.columns and not filtered_df['Status_Clean'].isna().all():
        status_counts = filtered_df['Status_Clean'].value_counts()
        pipeline_stages = [
            '0. Under Review/Intro Pending',
            '1. Scheduling Initial Call',
            'A. Initial Call Scheduled',
            'B. Early Discussions',
            'C. Ongoing Discussions',
            'D. Due Diligence Stage',
            'U.1. Agreement Executed',
            'Z. On Hold'
        ]
        
        for i, stage in enumerate(pipeline_stages):
            count = status_counts.get(stage, 0)
            with cols[i]:
                stage_label = stage.split(".")[-1].strip() if "." in stage else stage
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="font-size: 0.9rem;">{stage_label}</h3>
                    <p class="small-card">{count}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Create a pipeline flow visualization
    if 'Status_Clean' in filtered_df.columns and not filtered_df['Status_Clean'].isna().all():
        st.subheader("Recruitment Pipeline Flow")
        
        # Get counts for each stage
        status_counts = filtered_df['Status_Clean'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        
        # Define the pipeline stages in order
        pipeline_stages = [
            '0. Under Review/Intro Pending',
            '1. Scheduling Initial Call',
            'A. Initial Call Scheduled',
            'B. Early Discussions',
            'C. Ongoing Discussions',
            'D. Due Diligence Stage',
            'U.1. Agreement Executed'
        ]
        
        # Filter and order the status counts
        pipeline_data = status_counts[status_counts['Status'].isin(pipeline_stages)]
        if not pipeline_data.empty:
            pipeline_data['Order'] = pipeline_data['Status'].map({s: i for i, s in enumerate(pipeline_stages)})
            pipeline_data = pipeline_data.sort_values('Order')
            
            # Create a funnel chart
            fig = go.Figure(go.Funnel(
                y=pipeline_data['Status'].str.split('.').str[-1].str.strip(),
                x=pipeline_data['Count'],
                textposition="inside",
                textinfo="value+percent initial",
                marker={"color": ["#4e91dc", "#5a9be1", "#66a5e6", "#72afeb", "#7eb9f0", "#8ac3f5", "#96cdfa"]},
                connector={"line": {"color": "royalblue", "dash": "dot", "width": 3}}
            ))
            
            fig.update_layout(
                title="Recruitment Pipeline Funnel",
                height=400,
                margin=dict(t=50, b=0, l=150, r=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No pipeline data available")
    
    # Recent activity
    st.subheader("Recent Activity")
    if 'Last Activity Time_Date' in filtered_df.columns and not filtered_df['Last Activity Time_Date'].isna().all():
        recent_activity = filtered_df.sort_values('Last Activity Time_Date', ascending=False).head(5)
        if not recent_activity.empty:
            for _, row in recent_activity.iterrows():
                activity_date = row['Last Activity Time_Date'].strftime('%Y-%m-%d') if pd.notna(row['Last Activity Time_Date']) else "N/A"
                name = str(row['Full Name']) if pd.notna(row['Full Name']) else "Unknown"
                status = str(row['Recruit Status']) if pd.notna(row['Recruit Status']) else "N/A"
                practice = str(row['Practice Group / Sector']) if pd.notna(row['Practice Group / Sector']) else "N/A"
                
                st.markdown(f"**{name}** - {activity_date} - Status: {status} - Practice: {practice}")
    else:
        st.write("No recent activity data available")
    
    # Timeline of recruitment activity
    st.subheader("Recruitment Activity Over Time")
    if 'Last Activity Time_Date' in filtered_df.columns and not filtered_df['Last Activity Time_Date'].isna().all():
        # Group by month and count activities
        filtered_df['Month'] = filtered_df['Last Activity Time_Date'].dt.to_period('M')
        monthly_activity = filtered_df.groupby('Month').size().reset_index(name='Count')
        monthly_activity['Month'] = monthly_activity['Month'].dt.to_timestamp()
        
        # Create a line chart
        fig = px.line(
            monthly_activity,
            x='Month',
            y='Count',
            title='Monthly Recruitment Activity',
            markers=True,
            labels={'Month': 'Month', 'Count': 'Number of Activities'}
        )
        
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Number of Activities",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Practice Group Breakdown
    if 'Practice Group / Sector' in filtered_df.columns and not filtered_df['Practice Group / Sector'].isna().all():
        st.subheader("Practice Group Breakdown")
        
        practice_counts = filtered_df['Practice Group / Sector'].value_counts().reset_index()
        practice_counts.columns = ['Practice Group', 'Count']
        practice_counts = practice_counts.sort_values('Count', ascending=False).head(10)
        
        fig = px.bar(
            practice_counts,
            y='Practice Group',
            x='Count',
            title='Top 10 Practice Groups',
            orientation='h',
            color='Count',
            color_continuous_scale=px.colors.sequential.Blues,
            labels={'Practice Group': 'Practice Group', 'Count': 'Number of Recruits'}
        )
        
        fig.update_layout(
            height=500,
            margin=dict(l=300, r=20, t=50, b=50)
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Add minimal content to other tabs to avoid errors
with tabs[1]:
    st.header("Recruits by Source")
    if 'Recruit Source' in filtered_df.columns and not filtered_df['Recruit Source'].isna().all():
        # Process data for the chart
        source_counts = filtered_df['Recruit Source'].value_counts().reset_index()
        source_counts.columns = ['Source', 'Count']
        source_counts = source_counts[source_counts['Count'] > 0]
        
        if not source_counts.empty:
            # Create bar chart
            fig = px.bar(
                source_counts.head(10), 
                x='Source', 
                y='Count',
                title='Top 10 Recruit Sources',
                color='Count',
                color_continuous_scale=px.colors.sequential.Blues,
                labels={'Source': 'Recruit Source', 'Count': 'Number of Recruits'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show data table
            st.subheader("Detailed Source Breakdown")
            st.dataframe(source_counts, width=800)
        else:
            st.write("No source data available")
    else:
        st.write("No recruit source data available in the dataset")

with tabs[2]:
    st.header("Recruits by Status")
    if 'Status_Clean' in filtered_df.columns and not filtered_df['Status_Clean'].isna().all():
        # Process data for the chart
        status_counts = filtered_df['Status_Clean'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        
        if not status_counts.empty:
            # Create horizontal bar chart
            fig = px.bar(
                status_counts, 
                y='Status', 
                x='Count',
                title='Recruits by Status',
                orientation='h',
                color='Count',
                color_continuous_scale=px.colors.sequential.Blues,
                labels={'Status': 'Status', 'Count': 'Number of Recruits'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show data table
            st.subheader("Status Breakdown")
            st.dataframe(status_counts, width=800)
        else:
            st.write("No status data available")
    else:
        st.write("No recruit status data available in the dataset")

with tabs[3]:
    st.header("Referral Analysis")
    if 'Recruit Referral Details' in filtered_df.columns and not filtered_df['Recruit Referral Details'].isna().all():
        # Convert to string first
        filtered_df['Recruit Referral Details'] = filtered_df['Recruit Referral Details'].astype(str)
        referral_counts = filtered_df['Recruit Referral Details'].value_counts().reset_index()
        referral_counts.columns = ['Referral Source', 'Count']
        referral_counts = referral_counts[referral_counts['Count'] > 0]
        
        if not referral_counts.empty:
            # Create horizontal bar chart
            fig = px.bar(
                referral_counts.head(15), 
                y='Referral Source', 
                x='Count',
                title='Top 15 Referral Sources',
                orientation='h',
                color='Count',
                color_continuous_scale=px.colors.sequential.Blues,
                labels={'Referral Source': 'Referral Source', 'Count': 'Number of Recruits'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show data table
            st.subheader("All Referral Sources")
            st.dataframe(referral_counts, width=800)
        else:
            st.write("No referral details available")
    else:
        st.write("No recruit referral details available in the dataset")

with tabs[4]:
    st.header("Projected Business Analysis")
    if 'Estimated Book (Projected)_Value' in filtered_df.columns and not filtered_df['Estimated Book (Projected)_Value'].isna().all():
        # Filter out rows with missing book values
        book_df = filtered_df[filtered_df['Estimated Book (Projected)_Value'].notna()]
        
        if not book_df.empty:
            # Create histogram
            fig = px.histogram(
                book_df,
                x='Estimated Book (Projected)_Value',
                nbins=20,
                title='Distribution of Projected Book Values',
                labels={'Estimated Book (Projected)_Value': 'Projected Book Value ($)', 'count': 'Number of Recruits'},
                color_discrete_sequence=['#4e91dc']
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Basic metrics
            st.subheader("Book Value Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Projected", f"${book_df['Estimated Book (Projected)_Value'].sum():,.0f}")
            
            with col2:
                st.metric("Average Book", f"${book_df['Estimated Book (Projected)_Value'].mean():,.0f}")
            
            with col3:
                st.metric("Median Book", f"${book_df['Estimated Book (Projected)_Value'].median():,.0f}")
        else:
            st.write("No projected book value data available")
    else:
        st.write("No estimated book value data available in the dataset")

with tabs[5]:
    st.header("Timeline Analysis")
    st.info("This tab will show analysis of recruitment activity over time. Please ensure your data includes proper date fields.")

with tabs[6]:
    st.header("Geographic Distribution")
    st.info("This tab will show analysis of geographic distribution of recruits. Please ensure your data includes proper city/location fields.")

with tabs[7]:
    st.header("Conversion Funnel Analysis")
    st.info("This tab will show analysis of the recruitment funnel and conversion rates. Please ensure your data includes proper status fields.")

# Add a footer
st.markdown("---")

# Fix for the error - safely get the last update date
last_update_date = "Unknown"
if 'Last Activity Time_Date' in df.columns and not df['Last Activity Time_Date'].isna().all():
    try:
        # Get the max date using pandas method that handles NaN values
        last_date = df['Last Activity Time_Date'].max()
        if pd.notna(last_date):
            last_update_date = last_date.strftime('%Y-%m-%d')
    except Exception:
        pass  # If there's any error, keep the default "Unknown"

st.markdown(f"**Rimon Law Recruiting Dashboard** • Data Last Updated: {last_update_date}")
