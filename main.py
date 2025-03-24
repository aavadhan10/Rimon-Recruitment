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
        
        # Extract numeric values from status counts shown in your data
        if 'Recruit Status' in df.columns:
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
                df[field + '_Date'] = pd.to_datetime(df[field], errors='ignore')
        
        # Extract numeric values from monetary fields
        money_fields = ['Estimated Book (Projected)', 'Estimated Book (Conservative)', 'Estimated Book (Low)', 'Estimated Book (High)']
        for field in money_fields:
            if field in df.columns:
                df[field + '_Value'] = df[field].str.extract(r'\$\s*([\d,]+(?:\.\d+)?)')
                # Handle comma removal safely
                if df[field + '_Value'].notna().any():
                    df[field + '_Value'] = df[field + '_Value'].str.replace(',', '')
                # Convert to float with safe error handling
                df[field + '_Value'] = pd.to_numeric(df[field + '_Value'], errors='ignore')
        
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
if 'Status_Clean' in filtered_df.columns:
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
if 'Practice Group / Sector' in filtered_df.columns:
    practice_groups = filtered_df['Practice Group / Sector'].dropna().unique().tolist()
    selected_practice_groups = st.sidebar.multiselect(
        "Practice Group / Sector",
        options=practice_groups,
        default=[]
    )
    
    if selected_practice_groups:
        filtered_df = filtered_df[filtered_df['Practice Group / Sector'].isin(selected_practice_groups)]

# Recruit Source filter
if 'Recruit Source' in filtered_df.columns:
    sources = filtered_df['Recruit Source'].dropna().unique().tolist()
    selected_sources = st.sidebar.multiselect(
        "Recruit Source",
        options=sources,
        default=[]
    )
    
    if selected_sources:
        filtered_df = filtered_df[filtered_df['Recruit Source'].isin(selected_sources)]

# Book value range filter
if 'Estimated Book (Projected)_Value' in filtered_df.columns:
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
if 'City' in filtered_df.columns:
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
if 'Recruit Referral Details' in filtered_df.columns:
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
    active_recruits = filtered_df[filtered_df['Recruit Status'].notna() & ~filtered_df['Recruit Status'].str.contains('Z. On Hold', na=False)].shape[0]
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>Active Recruits</h3>
            <p>{}</p>
        </div>
        """.format(active_recruits), unsafe_allow_html=True)
    
    # On Hold
    on_hold = filtered_df[filtered_df['Recruit Status'].str.contains('Z. On Hold', na=False)].shape[0] if 'Recruit Status' in filtered_df.columns else 0
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>On Hold</h3>
            <p>{}</p>
        </div>
        """.format(on_hold), unsafe_allow_html=True)
    
    # Estimated Book Value (if available)
    if 'Estimated Book (Projected)_Value' in filtered_df.columns:
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
    
    if 'Status_Clean' in filtered_df.columns:
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
    if 'Status_Clean' in filtered_df.columns:
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
    
    # Recent activity
    st.subheader("Recent Activity")
    if 'Last Activity Time_Date' in filtered_df.columns and not filtered_df['Last Activity Time_Date'].isna().all():
        recent_activity = filtered_df.sort_values('Last Activity Time_Date', ascending=False).head(5)
        if not recent_activity.empty:
            for _, row in recent_activity.iterrows():
                activity_date = row['Last Activity Time_Date'].strftime('%Y-%m-%d') if pd.notna(row['Last Activity Time_Date']) else "N/A"
                name = row['Full Name'] if pd.notna(row['Full Name']) else "Unknown"
                status = row['Recruit Status'] if pd.notna(row['Recruit Status']) else "N/A"
                practice = row['Practice Group / Sector'] if pd.notna(row['Practice Group / Sector']) else "N/A"
                
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
    if 'Practice Group / Sector' in filtered_df.columns:
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

with tabs[1]:
    st.header("Recruits by Source")
    
    # Process data for the chart
    if 'Recruit Source' in filtered_df.columns:
        source_counts = filtered_df['Recruit Source'].value_counts().reset_index()
        source_counts.columns = ['Source', 'Count']
        source_counts = source_counts[source_counts['Count'] > 0]
        
        if not source_counts.empty:
            # Sort by count in descending order
            source_counts = source_counts.sort_values('Count', ascending=False)
            
            # Create two columns for visualizations
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Take top 10 sources for clarity
                top_sources = source_counts.head(10)
                
                # Create bar chart
                fig = px.bar(
                    top_sources, 
                    x='Source', 
                    y='Count',
                    title='Top 10 Recruit Sources',
                    color='Count',
                    color_continuous_scale=px.colors.sequential.Blues,
                    labels={'Source': 'Recruit Source', 'Count': 'Number of Recruits'}
                )
                
                fig.update_layout(
                    xaxis_title="Recruit Source",
                    yaxis_title="Number of Recruits",
                    yaxis=dict(tickmode='linear'),
                    height=500,
                    margin=dict(t=50, b=100)
                )
                
                # Rotate x-axis labels for better readability
                fig.update_xaxes(tickangle=45)
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Create a pie chart of top 5 sources
                top5_sources = source_counts.head(5)
                other_count = source_counts.iloc[5:]['Count'].sum()
                
                # Add "Other" category
                pie_data = pd.concat([
                    top5_sources,
                    pd.DataFrame({'Source': ['Other'], 'Count': [other_count]})
                ])
                
                # Calculate percentages
                total = pie_data['Count'].sum()
                pie_data['Percentage'] = (pie_data['Count'] / total * 100).round(1)
                pie_data['Label'] = pie_data['Source'] + ' (' + pie_data['Percentage'].astype(str) + '%)'
                
                fig_pie = px.pie(
                    pie_data,
                    names='Label',
                    values='Count',
                    title='Source Distribution',
                    hole=0.4,
                    color_discrete_sequence=px.colors.sequential.Blues
                )
                
                fig_pie.update_traces(textposition='inside', textinfo='percent')
                
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Comparison of source effectiveness
            st.subheader("Source Effectiveness")
            
            # Calculate conversion metrics by source
            if 'Status_Clean' in filtered_df.columns:
                # Consider "converted" recruits as those in late stages
                late_stages = ['C. Ongoing Discussions', 'D. Due Diligence Stage', 'U.1. Agreement Executed']
                
                # Group by source and count total and converted
                source_effectiveness = filtered_df.groupby('Recruit Source').agg(
                    Total=('Recruit Source', 'size'),
                    Converted=('Status_Clean', lambda x: sum(x.isin(late_stages)))
                ).reset_index()
                
                # Calculate conversion rate
                source_effectiveness['Conversion Rate'] = (source_effectiveness['Converted'] / source_effectiveness['Total'] * 100).round(1)
                
                # Sort by conversion rate and filter for sources with at least 5 recruits
                source_effectiveness = source_effectiveness[source_effectiveness['Total'] >= 5]
                source_effectiveness = source_effectiveness.sort_values('Conversion Rate', ascending=False).head(10)
                
                # Create a horizontal bar chart
                fig = px.bar(
                    source_effectiveness,
                    y='Recruit Source',
                    x='Conversion Rate',
                    title='Top 10 Sources by Conversion Rate (min. 5 recruits)',
                    orientation='h',
                    color='Conversion Rate',
                    color_continuous_scale=px.colors.sequential.Blues,
                    text='Conversion Rate',
                    labels={'Recruit Source': 'Source', 'Conversion Rate': 'Conversion Rate (%)'}
                )
                
                fig.update_layout(
                    height=500,
                    margin=dict(l=250, r=20, t=50, b=50)
                )
                
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Source trends over time
            st.subheader("Source Trends Over Time")
            
            if 'Last Activity Time_Date' in filtered_df.columns:
                # Group by month and source
                filtered_df['Month'] = filtered_df['Last Activity Time_Date'].dt.to_period('M')
                
                # Get top 5 sources
                top_sources = source_counts.head(5)['Source'].tolist()
                
                # Filter for top sources only
                source_trend_data = filtered_df[filtered_df['Recruit Source'].isin(top_sources)]
                
                # Group by month and source
                source_trends = source_trend_data.groupby(['Month', 'Recruit Source']).size().reset_index(name='Count')
                source_trends['Month'] = source_trends['Month'].dt.to_timestamp()
                
                # Create a line chart
                fig = px.line(
                    source_trends,
                    x='Month',
                    y='Count',
                    color='Recruit Source',
                    title='Recruitment Source Trends Over Time',
                    markers=True,
                    labels={'Month': 'Month', 'Count': 'Number of Recruits', 'Recruit Source': 'Source'}
                )
                
                fig.update_layout(
                    xaxis_title="Month",
                    yaxis_title="Number of Recruits",
                    height=400,
                    legend_title="Source"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed breakdown table
            st.subheader("Detailed Source Breakdown")
            
            # Add total and percentage to the source counts
            total_recruits = source_counts['Count'].sum()
            source_counts['Percentage'] = (source_counts['Count'] / total_recruits * 100).round(2)
            source_counts['Percentage'] = source_counts['Percentage'].astype(str) + '%'
            
            st.dataframe(source_counts, width=800)
        else:
            st.write("No source data available")
    else:
        st.write("No recruit source data available in the dataset")

with tabs[2]:
    st.header("Recruits by Status")
    
    # Process data for the chart
    if 'Recruit Status' in filtered_df.columns:
        status_counts = filtered_df['Recruit Status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        status_counts = status_counts[status_counts['Count'] > 0]
        
        if not status_counts.empty:
            # Clean status names to make them more readable
            status_counts['Status_Clean'] = status_counts['Status'].str.extract(r'(.*?)\s*\(', expand=False).str.strip()
            
            # Sort by the order in the recruitment pipeline
            status_order = [
                '0. Under Review/Intro Pending',
                '1. Scheduling Initial Call',
                'A. Initial Call Scheduled',
                'B. Early Discussions',
                'C. Ongoing Discussions',
                'D. Due Diligence Stage',
                'U.1. Agreement Executed',
                'Z. On Hold'
            ]
            
            # Map for ordering
            status_map = {status: i for i, status in enumerate(status_order)}
            
            # Apply the ordering if the status is in our predefined list
            status_counts['Order'] = status_counts['Status_Clean'].map(lambda x: status_map.get(x, 999))
            status_counts = status_counts.sort_values('Order')
            
            # Create two columns for visualizations
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # Create horizontal bar chart
                fig = px.bar(
                    status_counts, 
                    y='Status_Clean', 
                    x='Count',
                    title='Recruits by Status',
                    orientation='h',
                    color='Count',
                    color_continuous_scale=px.colors.sequential.Blues,
                    labels={'Status_Clean': 'Status', 'Count': 'Number of Recruits'}
                )
                
                fig.update_layout(
                    xaxis_title="Number of Recruits",
                    yaxis_title="Status",
                    height=500,
                    margin=dict(l=150, r=20, t=50, b=50)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Calculate percentages for each status
                total = status_counts['Count'].sum()
                status_counts['Percentage'] = (status_counts['Count'] / total * 100).round(2)
                status_counts['Display'] = status_counts['Status_Clean'] + ' (' + status_counts['Percentage'].astype(str) + '%)'
                
                # Create a pie chart showing the distribution
                fig_pie = px.pie(
                    status_counts, 
                    values='Count', 
                    names='Display',
                    title='Status Distribution (%)',
                    color_discrete_sequence=px.colors.sequential.Blues_r,
                    hole=0.4
                )
                
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_pie.update_layout(margin=dict(t=50, b=50))
                
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Status by practice group
            st.subheader("Status by Practice Group")
            
            if 'Practice Group / Sector' in filtered_df.columns:
                # Get top practice groups
                top_practices = filtered_df['Practice Group / Sector'].value_counts().head(5).index.tolist()
                
                # Filter data
                practice_status_data = filtered_df[filtered_df['Practice Group / Sector'].isin(top_practices)]
                
                # Create a contingency table
                practice_status = pd.crosstab(
                    practice_status_data['Practice Group / Sector'],
                    practice_status_data['Status_Clean']
                )
                
                # Reorder columns based on status order
                ordered_columns = [col for col in status_order if col in practice_status.columns]
                practice_status = practice_status[ordered_columns]
                
                # Create a stacked bar chart
                fig = px.bar(
                    practice_status, 
                    barmode='stack',
                    title='Recruit Status by Top 5 Practice Groups',
                    color_discrete_sequence=px.colors.sequential.Blues_r,
                    labels={'value': 'Number of Recruits', 'index': 'Practice Group'}
