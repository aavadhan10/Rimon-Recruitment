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
            df['Status_Count'] = pd.to_numeric(df['Status_Count'], errors='coerce')
            
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
                df[field + '_Value'] = df[field].str.extract(r'\$\s*([\d,]+(?:\.\d+)?)')
                df[field + '_Value'] = df[field + '_Value'].str.replace(',', '').astype(float, errors='coerce')
        
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
            )
            
            fig.update_layout(
                xaxis_title="Practice Group",
                yaxis_title="Number of Recruits",
                legend_title="Status",
                height=500,
                margin=dict(l=0, r=0, t=50, b=100)
            )
            
            # Rotate x-axis labels for better readability
            fig.update_xaxes(tickangle=45)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Status transition analysis (if we had time data)
            st.subheader("Status Transition Analysis")
            
            if 'Last Activity Time_Date' in filtered_df.columns:
                # This would typically require historical data of status changes
                # Since we don't have that, we'll simulate with a placeholder visualization
                
                st.info("""
                This section requires historical status change data to show how recruits move through the pipeline over time.
                Consider adding timestamp data for status changes to enable this analysis.
                """)
                
                # Create sample data for illustration
                stages = ['Initial Contact', 'Screening', 'Interview', 'Due Diligence', 'Offer', 'Accepted']
                avg_days = [0, 7, 21, 35, 50, 65]
                std_days = [0, 3, 7, 10, 8, 5]
                
                # Create dataframe
                timeline_data = pd.DataFrame({
                    'Stage': stages,
                    'Average Days': avg_days,
                    'Lower': [a - s for a, s in zip(avg_days, std_days)],
                    'Upper': [a + s for a, s in zip(avg_days, std_days)]
                })
                
                # Create a line chart
                fig = go.Figure()
                
                # Add the main line
                fig.add_trace(go.Scatter(
                    x=timeline_data['Stage'],
                    y=timeline_data['Average Days'],
                    mode='lines+markers',
                    name='Average Days',
                    line=dict(color='royalblue', width=3),
                    marker=dict(size=10)
                ))
                
                # Add the range
                fig.add_trace(go.Scatter(
                    x=timeline_data['Stage'],
                    y=timeline_data['Upper'],
                    mode='lines',
                    name='Upper Bound',
                    line=dict(width=0),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=timeline_data['Stage'],
                    y=timeline_data['Lower'],
                    mode='lines',
                    name='Lower Bound',
                    line=dict(width=0),
                    fillcolor='rgba(68, 138, 255, 0.2)',
                    fill='tonexty',
                    showlegend=False
                ))
                
                fig.update_layout(
                    title='Average Time Spent in Each Recruitment Stage (Sample Data)',
                    xaxis_title='Recruitment Stage',
                    yaxis_title='Days from Initial Contact',
                    height=400,
                    margin=dict(l=0, r=0, t=50, b=50)
                )
                
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No status data available")
else:
    st.write("No recruit status data available in the dataset")

with tabs[3]:
    st.header("Referral Analysis")
    
    # Process data for the referral analysis
    if 'Recruit Referral Details' in filtered_df.columns:
        referral_counts = filtered_df['Recruit Referral Details'].value_counts().reset_index()
        referral_counts.columns = ['Referral Source', 'Count']
        referral_counts = referral_counts[referral_counts['Count'] > 0]
        
        if not referral_counts.empty:
            # Create overview metrics
            st.subheader("Referral Overview")
            
            col1, col2, col3 = st.columns(3)
            
            # Total referrals
            total_referrals = referral_counts['Count'].sum()
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <h3>Total Referrals</h3>
                    <p>{}</p>
                </div>
                """.format(total_referrals), unsafe_allow_html=True)
            
            # Unique referral sources
            unique_sources = len(referral_counts)
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <h3>Unique Sources</h3>
                    <p>{}</p>
                </div>
                """.format(unique_sources), unsafe_allow_html=True)
            
            # Average referrals per source
            avg_per_source = (total_referrals / unique_sources).round(1)
            with col3:
                st.markdown("""
                <div class="metric-card">
                    <h3>Avg per Source</h3>
                    <p>{}</p>
                </div>
                """.format(avg_per_source), unsafe_allow_html=True)
            
            # Take top 15 referral sources
            top_referrals = referral_counts.head(15)
            
            # Create horizontal bar chart
            fig = px.bar(
                top_referrals, 
                y='Referral Source', 
                x='Count',
                title='Top 15 Referral Sources',
                orientation='h',
                color='Count',
                color_continuous_scale=px.colors.sequential.Blues,
                labels={'Referral Source': 'Referral Source', 'Count': 'Number of Recruits'}
            )
            
            fig.update_layout(
                xaxis_title="Number of Recruits",
                yaxis_title="Referral Source",
                height=600,
                margin=dict(l=250, r=20, t=50, b=50)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Referral quality analysis
            st.subheader("Referral Quality Analysis")
            
            if 'Status_Clean' in filtered_df.columns and 'Estimated Book (Projected)_Value' in filtered_df.columns:
                # Consider "high quality" referrals as those in later stages with high book values
                late_stages = ['C. Ongoing Discussions', 'D. Due Diligence Stage', 'U.1. Agreement Executed']
                
                # Add quality indicators to the dataframe
                filtered_df['Is_Late_Stage'] = filtered_df['Status_Clean'].isin(late_stages)
                filtered_df['Has_Book_Value'] = filtered_df['Estimated Book (Projected)_Value'].notna()
                
                # Group by referral source and calculate metrics
                referral_quality = filtered_df.groupby('Recruit Referral Details').agg(
                    Total=('Recruit Referral Details', 'size'),
                    Late_Stage=('Is_Late_Stage', 'sum'),
                    Avg_Book_Value=('Estimated Book (Projected)_Value', 'mean')
                ).reset_index()
                
                # Calculate rate and filter for sources with at least 3 referrals
                referral_quality['Late_Stage_Rate'] = (referral_quality['Late_Stage'] / referral_quality['Total'] * 100).round(1)
                referral_quality = referral_quality[referral_quality['Total'] >= 3]
                
                # Sort by quality metrics
                top_quality_sources = referral_quality.sort_values('Late_Stage_Rate', ascending=False).head(10)
                
                # Create a bubble chart
                fig = px.scatter(
                    top_quality_sources,
                    x='Late_Stage_Rate',
                    y='Avg_Book_Value',
                    size='Total',
                    color='Late_Stage_Rate',
                    hover_name='Recruit Referral Details',
                    color_continuous_scale=px.colors.sequential.Blues,
                    title='Top Referral Sources by Quality',
                    labels={
                        'Late_Stage_Rate': 'Late Stage Conversion Rate (%)',
                        'Avg_Book_Value': 'Average Book Value ($)',
                        'Total': 'Total Referrals'
                    }
                )
                
                fig.update_layout(
                    height=500,
                    margin=dict(l=0, r=0, t=50, b=0)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Referral network visualization
            st.subheader("Referral Network")
            
            # For this example, we'll create a simple visualization of how referrals are connected
            # In a real implementation, you'd need data about the relationships between referrers
            
            st.info("""
            This section could show a network graph of how referrals are connected to each other.
            Consider tracking relationships between referral sources to enable this visualization.
            """)
            
            # Search functionality for referrals
            st.subheader("Search Referral Sources")
            search_term = st.text_input("Search for a referral source:")
            
            if search_term:
                filtered_referrals = referral_counts[referral_counts['Referral Source'].str.contains(search_term, case=False, na=False)]
                if not filtered_referrals.empty:
                    st.dataframe(filtered_referrals, width=800)
                else:
                    st.write("No matching referral sources found.")
            else:
                # Show all referral sources in a table
                st.dataframe(referral_counts, width=800)
        else:
            st.write("No referral details available")
    else:
        st.write("No recruit referral details available in the dataset")

with tabs[4]:
    st.header("Projected Business Analysis")
    
    # Process data for the projected business analysis
    if 'Estimated Book (Projected)_Value' in filtered_df.columns:
        # Filter out rows with missing book values
        book_df = filtered_df[filtered_df['Estimated Book (Projected)_Value'].notna()]
        
        if not book_df.empty:
            # Create overview metrics
            st.subheader("Book Value Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            # Total projected business
            total_projected = book_df['Estimated Book (Projected)_Value'].sum()
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <h3>Total Projected</h3>
                    <p>${:,.0f}</p>
                </div>
                """.format(total_projected), unsafe_allow_html=True)
            
            # Average book value
            avg_book = book_df['Estimated Book (Projected)_Value'].mean()
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <h3>Average Book</h3>
                    <p>${:,.0f}</p>
                </div>
                """.format(avg_book), unsafe_allow_html=True)
            
            # Median book value
            median_book = book_df['Estimated Book (Projected)_Value'].median()
            with col3:
                st.markdown("""
                <div class="metric-card">
                    <h3>Median Book</h3>
                    <p>${:,.0f}</p>
                </div>
                """.format(median_book), unsafe_allow_html=True)
            
            # Candidates with book value
            candidates_with_book = len(book_df)
            with col4:
                st.markdown("""
                <div class="metric-card">
                    <h3>Candidates with Book</h3>
                    <p>{}</p>
                </div>
                """.format(candidates_with_book), unsafe_allow_html=True)
            
            # Create a histogram of book values
            fig = px.histogram(
                book_df,
                x='Estimated Book (Projected)_Value',
                nbins=20,
                title='Distribution of Projected Book Values',
                labels={'Estimated Book (Projected)_Value': 'Projected Book Value ($)', 'count': 'Number of Recruits'},
                color_discrete_sequence=['#4e91dc']
            )
            
            fig.update_layout(
                xaxis_title="Projected Book Value ($)",
                yaxis_title="Number of Recruits",
                height=400
            )
            
            # Add a vertical line for the average
            fig.add_vline(x=avg_book, line_dash="dash", line_color="red", annotation_text=f"Avg: ${avg_book:,.0f}")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Create a box plot to show the distribution by status
            if 'Status_Clean' in book_df.columns:
                active_statuses = [s for s in status_order if s != 'Z. On Hold']
                status_book_df = book_df[book_df['Status_Clean'].isin(active_statuses)]
                
                if not status_book_df.empty:
                    fig = px.box(
                        status_book_df,
                        x='Status_Clean',
                        y='Estimated Book (Projected)_Value',
                        title='Book Value Distribution by Status',
                        labels={
                            'Status_Clean': 'Status',
                            'Estimated Book (Projected)_Value': 'Projected Book Value ($)'
                        },
                        color='Status_Clean',
                        color_discrete_sequence=px.colors.sequential.Blues_r
                    )
                    
                    fig.update_layout(
                        xaxis_title="Status",
                        yaxis_title="Projected Book Value ($)",
                        height=500,
                        margin=dict(l=0, r=0, t=50, b=100)
                    )
                    
                    # Rotate x-axis labels for better readability
                    fig.update_xaxes(tickangle=45)
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Breakdown by practice group if available
            if 'Practice Group / Sector' in book_df.columns:
                st.subheader("Book Value by Practice Group")
                
                practice_totals = book_df.groupby('Practice Group / Sector')['Estimated Book (Projected)_Value'].agg(['sum', 'count', 'mean']).reset_index()
                practice_totals = practice_totals.sort_values('sum', ascending=False)
                practice_totals.columns = ['Practice Group', 'Total Projected ($)', 'Number of Recruits', 'Average Projected ($)']
                
                # Format currency columns for display
                practice_totals_display = practice_totals.copy()
                practice_totals_display['Total Projected ($)'] = practice_totals_display['Total Projected ($)'].apply(lambda x: f"${x:,.0f}")
                practice_totals_display['Average Projected ($)'] = practice_totals_display['Average Projected ($)'].apply(lambda x: f"${x:,.0f}")
                
                st.dataframe(practice_totals_display, width=800)
                
                # Visual representation
                top_practices = practice_totals.head(10)
                
                # Create a grouped bar chart
                fig = go.Figure()
                
                # Add total bar
                fig.add_trace(go.Bar(
                    x=top_practices['Practice Group'],
                    y=top_practices['Total Projected ($)'],
                    name='Total Projected ($)',
                    marker_color='royalblue'
                ))
                
                # Add average bar
                fig.add_trace(go.Bar(
                    x=top_practices['Practice Group'],
                    y=top_practices['Average Projected ($)'],
                    name='Average Projected ($)',
                    marker_color='lightblue'
                ))
                
                fig.update_layout(
                    title='Top 10 Practice Groups by Projected Business',
                    xaxis_title="Practice Group",
                    yaxis_title="Amount ($)",
                    barmode='group',
                    height=500,
                    margin=dict(t=50, b=150),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                # Rotate x-axis labels for better readability
                fig.update_xaxes(tickangle=45)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Create a pie chart for total book value by practice
                fig_pie = px.pie(
                    top_practices,
                    values='Total Projected ($)',
                    names='Practice Group',
                    title='Distribution of Total Book Value by Practice Group',
                    hole=0.4,
                    color_discrete_sequence=px.colors.sequential.Blues
                )
                
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_pie.update_layout(height=500)
                
                st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.write("No projected book value data available")
    else:
        st.write("No estimated book value data available in the dataset")

with tabs[5]:
    st.header("Timeline Analysis")
    
    if 'Last Activity Time_Date' in filtered_df.columns and not filtered_df['Last Activity Time_Date'].isna().all():
        # Create overview metrics
        st.subheader("Activity Trends")
        
        # Group by month and year
        filtered_df['Year'] = filtered_df['Last Activity Time_Date'].dt.year
        filtered_df['Month'] = filtered_df['Last Activity Time_Date'].dt.month
        filtered_df['YearMonth'] = filtered_df['Last Activity Time_Date'].dt.to_period('M')
        
        # Activity by month
        monthly_activity = filtered_df.groupby('YearMonth').size().reset_index(name='Count')
        monthly_activity['YearMonth'] = monthly_activity['YearMonth'].dt.to_timestamp()
        
        # Create a line chart
        fig = px.line(
            monthly_activity,
            x='YearMonth',
            y='Count',
            title='Monthly Recruitment Activity',
            markers=True,
            labels={'YearMonth': 'Month', 'Count': 'Number of Activities'}
        )
        
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Number of Activities",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Status changes over time
        if 'Status_Clean' in filtered_df.columns:
            st.subheader("Status Composition Over Time")
            
            # Group by month and status
            status_over_time = filtered_df.groupby(['YearMonth', 'Status_Clean']).size().reset_index(name='Count')
            status_over_time['YearMonth'] = status_over_time['YearMonth'].dt.to_timestamp()
            
            # Only include the last 12 months
            last_12_months = sorted(status_over_time['YearMonth'].unique())[-12:]
            status_over_time = status_over_time[status_over_time['YearMonth'].isin(last_12_months)]
            
            # Create a stacked area chart
            fig = px.area(
                status_over_time,
                x='YearMonth',
                y='Count',
                color='Status_Clean',
                title='Recruit Status Composition Over Time',
                labels={'YearMonth': 'Month', 'Count': 'Number of Recruits', 'Status_Clean': 'Status'}
            )
            
            fig.update_layout(
                xaxis_title="Month",
                yaxis_title="Number of Recruits",
                height=500,
                legend_title="Status"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Activity by day of week
        st.subheader("Activity by Day of Week")
        
        # Add day of week
        filtered_df['DayOfWeek'] = filtered_df['Last Activity Time_Date'].dt.day_name()
        
        # Count activities by day
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_activity = filtered_df['DayOfWeek'].value_counts().reindex(day_order).reset_index()
        day_activity.columns = ['Day', 'Count']
        
        # Create a bar chart
        fig = px.bar(
            day_activity,
            x='Day',
            y='Count',
            title='Activity by Day of Week',
            color='Count',
            color_continuous_scale=px.colors.sequential.Blues,
            labels={'Day': 'Day of Week', 'Count': 'Number of Activities'}
        )
        
        fig.update_layout(
            xaxis_title="Day of Week",
            yaxis_title="Number of Activities",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Activity by hour of day
        st.subheader("Activity by Hour of Day")
        
        # Extract hour
        if 'Last Activity Time' in filtered_df.columns:
            try:
                # Try to extract time component
                filtered_df['Hour'] = filtered_df['Last Activity Time'].str.extract(r'(\d+):\d+ (AM|PM)')
                filtered_df['AM_PM'] = filtered_df['Last Activity Time'].str.extract(r'\d+:\d+ (AM|PM)')
                
                # Convert hour to 24-hour format
                filtered_df['Hour'] = pd.to_numeric(filtered_df['Hour'], errors='coerce')
                filtered_df.loc[(filtered_df['AM_PM'] == 'PM') & (filtered_df['Hour'] < 12), 'Hour'] += 12
                filtered_df.loc[(filtered_df['AM_PM'] == 'AM') & (filtered_df['Hour'] == 12), 'Hour'] = 0
                
                # Count activities by hour
                hour_activity = filtered_df['Hour'].value_counts().sort_index().reset_index()
                hour_activity.columns = ['Hour', 'Count']
                
                # Fill in missing hours
                all_hours = pd.DataFrame({'Hour': range(24)})
                hour_activity = pd.merge(all_hours, hour_activity, on='Hour', how='left').fillna(0)
                
                # Create a bar chart
                fig = px.bar(
                    hour_activity,
                    x='Hour',
                    y='Count',
                    title='Activity by Hour of Day',
                    color='Count',
                    color_continuous_scale=px.colors.sequential.Blues,
                    labels={'Hour': 'Hour of Day (24-hour format)', 'Count': 'Number of Activities'}
                )
                
                fig.update_layout(
                    xaxis_title="Hour of Day",
                    yaxis_title="Number of Activities",
                    height=400,
                    xaxis=dict(tickmode='linear', tick0=0, dtick=1)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.write(f"Could not parse time information: {e}")
    else:
        st.write("No activity timeline data available")

with tabs[6]:
    st.header("Geographic Distribution")
    
    if 'City' in filtered_df.columns and not filtered_df['City'].isna().all():
        # Clean city data (remove any count indicators in parentheses)
        filtered_df['City_Clean'] = filtered_df['City'].str.extract(r'(.*?)\s*\(', expand=False).str.strip()
        
        # City distribution
        city_counts = filtered_df['City_Clean'].value_counts().reset_index()
        city_counts.columns = ['City', 'Count']
        
        # Filter out empty or invalid cities
        city_counts = city_counts[city_counts['City'].notna() & (city_counts['City'] != '-') & (city_counts['City'] != '')]
        
        if not city_counts.empty:
            # Take top cities
            top_cities = city_counts.head(15)
            
            # Create a bar chart
            fig = px.bar(
                top_cities,
                y='City',
                x='Count',
                title='Top 15 Cities by Number of Recruits',
                orientation='h',
                color='Count',
                color_continuous_scale=px.colors.sequential.Blues,
                labels={'City': 'City', 'Count': 'Number of Recruits'}
            )
            
            fig.update_layout(
                xaxis_title="Number of Recruits",
                yaxis_title="City",
                height=500,
                margin=dict(l=150, r=20, t=50, b=50)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # City by status
            if 'Status_Clean' in filtered_df.columns:
                st.subheader("Recruitment Status by City")
                
                # Get top 5 cities
                top5_cities = city_counts.head(5)['City'].tolist()
                
                # Filter data for top cities
                city_status_data = filtered_df[filtered_df['City_Clean'].isin(top5_cities)]
                
                # Create a contingency table
                city_status = pd.crosstab(
                    city_status_data['City_Clean'],
                    city_status_data['Status_Clean']
                )
                
                # Create a stacked bar chart
                fig = px.bar(
                    city_status,
                    barmode='stack',
                    title='Recruit Status by Top 5 Cities',
                    color_discrete_sequence=px.colors.sequential.Blues_r,
                    labels={'value': 'Number of Recruits', 'index': 'City'}
                )
                
                fig.update_layout(
                    xaxis_title="City",
                    yaxis_title="Number of Recruits",
                    legend_title="Status",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # City by practice area
            if 'Practice Group / Sector' in filtered_df.columns:
                st.subheader("Practice Groups by City")
                
                # Get top practices
                top_practices = filtered_df['Practice Group / Sector'].value_counts().head(3).index.tolist()
                
                # Filter data
                city_practice_data = filtered_df[
                    (filtered_df['City_Clean'].isin(top5_cities)) & 
                    (filtered_df['Practice Group / Sector'].isin(top_practices))
                ]
                
                # Create a contingency table
                city_practice = pd.crosstab(
                    city_practice_data['City_Clean'],
                    city_practice_data['Practice Group / Sector']
                )
                
                # Create a grouped bar chart
                fig = px.bar(
                    city_practice,
                    barmode='group',
                    title='Top 3 Practice Groups in Top 5 Cities',
                    color_discrete_sequence=px.colors.sequential.Blues_r,
                    labels={'value': 'Number of Recruits', 'index': 'City'}
                )
                
                fig.update_layout(
                    xaxis_title="City",
                    yaxis_title="Number of Recruits",
                    legend_title="Practice Group",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed table of cities
            st.subheader("All Cities")
            st.dataframe(city_counts, width=800)
        else:
            st.write("No valid city data available")
    else:
        st.write("No geographic data available in the dataset")

with tabs[7]:
    st.header("Conversion Funnel Analysis")
    
    if 'Status_Clean' in filtered_df.columns:
        st.subheader("Recruitment Funnel")
        
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
        
        # Count recruits in each stage
        stage_counts = filtered_df['Status_Clean'].value_counts().reindex(pipeline_stages).fillna(0).reset_index()
        stage_counts.columns = ['Stage', 'Count']
        
        # Calculate conversion rates
        if len(stage_counts) > 0 and stage_counts['Count'].sum() > 0:
            initial_count = stage_counts.iloc[0]['Count']
            stage_counts['Conversion Rate'] = (stage_counts['Count'] / initial_count * 100).round(1)
            stage_counts['Conversion Rate'] = stage_counts['Conversion Rate'].astype(str) + '%'
            
            # Create a funnel chart
            simplified_stages = [s.split('.')[-1].strip() for s in stage_counts['Stage']]
            
            fig = go.Figure(go.Funnel(
                y=simplified_stages,
                x=stage_counts['Count'],
                textposition="inside",
                textinfo="value+percent initial",
                marker={"color": ["#4e91dc", "#5a9be1", "#66a5e6", "#72afeb", "#7eb9f0", "#8ac3f5", "#96cdfa"]},
                connector={"line": {"color": "royalblue", "dash": "dot", "width": 3}}
            ))
            
            fig.update_layout(
                title="Recruitment Pipeline Funnel",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show the data in a table
            st.subheader("Pipeline Stage Metrics")
            st.dataframe(stage_counts, width=800)
        
        # Conversion metrics by different factors
        st.subheader("Conversion Analysis")
        
        # Define what qualifies as conversion (late stage or executed agreement)
        late_stages = ['C. Ongoing Discussions', 'D. Due Diligence Stage', 'U.1. Agreement Executed']
        filtered_df['Converted'] = filtered_df['Status_Clean'].isin(late_stages)
        
        # Conversion factors to analyze
        analysis_factors = []
        
        if 'Recruit Source' in filtered_df.columns:
            analysis_factors.append('Recruit Source')
        
        if 'Practice Group / Sector' in filtered_df.columns:
            analysis_factors.append('Practice Group / Sector')
        
        if 'City_Clean' in filtered_df.columns:
            analysis_factors.append('City_Clean')
        
        if 'Recruit Referral Details' in filtered_df.columns:
            analysis_factors.append('Recruit Referral Details')
        
        # Select a factor to analyze
        if analysis_factors:
            selected_factor = st.selectbox("Select a factor to analyze conversion rates:", analysis_factors)
            
            # Calculate conversion by the selected factor
            conversion_by_factor = filtered_df.groupby(selected_factor).agg(
                Total=('Converted', 'size'),
                Converted=('Converted', 'sum')
            ).reset_index()
            
            # Calculate conversion rate
            conversion_by_factor['Conversion Rate'] = (conversion_by_factor['Converted'] / conversion_by_factor['Total'] * 100).round(1)
            
            # Filter for factors with at least 3 entries
            conversion_by_factor = conversion_by_factor[conversion_by_factor['Total'] >= 3]
            
            # Sort by conversion rate and take top 10
            top_conversion = conversion_by_factor.sort_values('Conversion Rate', ascending=False).head(10)
            
            # Create a horizontal bar chart
            fig = px.bar(
                top_conversion,
                y=selected_factor,
                x='Conversion Rate',
                title=f'Top 10 {selected_factor} by Conversion Rate (min. 3 entries)',
                orientation='h',
                color='Conversion Rate',
                color_continuous_scale=px.colors.sequential.Blues,
                text='Conversion Rate',
                labels={selected_factor: selected_factor, 'Conversion Rate': 'Conversion Rate (%)'}
            )
            
            fig.update_layout(
                height=500,
                margin=dict(l=200, r=20, t=50, b=50)
            )
            
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show the data in a table
            st.dataframe(top_conversion, width=800)
        else:
            st.write("No suitable factors available for conversion analysis")
        
        # Time to conversion analysis
        st.subheader("Time to Conversion Analysis")
        
        if 'Created Time_Date' in filtered_df.columns and 'Last Activity Time_Date' in filtered_df.columns:
            # Calculate days in pipeline
            filtered_df['Days_in_Pipeline'] = (filtered_df['Last Activity Time_Date'] - filtered_df['Created Time_Date']).dt.days
            
            # Create a box plot by status
            valid_time_df = filtered_df[(filtered_df['Days_in_Pipeline'] >= 0) & (filtered_df['Days_in_Pipeline'].notna())]
            
            if not valid_time_df.empty:
                fig = px.box(
                    valid_time_df,
                    x='Status_Clean',
                    y='Days_in_Pipeline',
                    title='Days in Pipeline by Status',
                    color='Status_Clean',
                    color_discrete_sequence=px.colors.sequential.Blues_r,
                    labels={'Status_Clean': 'Status', 'Days_in_Pipeline': 'Days in Pipeline'}
                )
                
                fig.update_layout(
                    xaxis_title="Status",
                    yaxis_title="Days in Pipeline",
                    height=500,
                    margin=dict(l=0, r=0, t=50, b=100)
                )
                
                # Rotate x-axis labels for better readability
                fig.update_xaxes(tickangle=45)
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No valid time data available for pipeline analysis")
        else:
            st.write("Date information needed for time-to-conversion analysis is not available")
    else:
        st.write("Status information needed for conversion analysis is not available")

# Add a footer
st.markdown("---")
st.markdown("**Rimon Law Recruiting Dashboard** • Data Last Updated: {}".format(
    df['Last Activity Time'].max() if 'Last Activity Time' in df.columns else "Unknown"
))

