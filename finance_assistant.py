import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from textblob import TextBlob
import csv
import io
import os


# Database Setup
def init_db():
    with sqlite3.connect('finance.db') as conn:
        c = conn.cursor()
        c.execute(
            '''CREATE TABLE IF NOT EXISTS expenses (id INTEGER PRIMARY KEY, date TEXT, amount REAL, category TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS income (id INTEGER PRIMARY KEY, date TEXT, amount REAL, source TEXT)''')
        c.execute(
            '''CREATE TABLE IF NOT EXISTS sentiment (id INTEGER PRIMARY KEY, date TEXT, sentiment_score REAL, source TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS budget (id INTEGER PRIMARY KEY, month TEXT, budget_limit REAL)''')
        c.execute(
            '''CREATE TABLE IF NOT EXISTS savings_goals (id INTEGER PRIMARY KEY, goal_name TEXT, target_amount REAL, current_amount REAL, target_date TEXT)''')
        conn.commit()


def add_expense(date, amount, category):
    with sqlite3.connect('finance.db') as conn:
        c = conn.cursor()
        c.execute("INSERT INTO expenses (date, amount, category) VALUES (?, ?, ?)", (date, amount, category))
        conn.commit()


def add_income(date, amount, source):
    with sqlite3.connect('finance.db') as conn:
        c = conn.cursor()
        c.execute("INSERT INTO income (date, amount, source) VALUES (?, ?, ?)", (date, amount, source))
        conn.commit()


def add_sentiment(date, sentiment_score, source):
    with sqlite3.connect('finance.db') as conn:
        c = conn.cursor()
        c.execute("INSERT INTO sentiment (date, sentiment_score, source) VALUES (?, ?, ?)",
                  (date, sentiment_score, source))
        conn.commit()


def set_budget(month, budget_limit):
    with sqlite3.connect('finance.db') as conn:
        c = conn.cursor()
        c.execute(
            "INSERT OR REPLACE INTO budget (id, month, budget_limit) VALUES ((SELECT id FROM budget WHERE month = ?), ?, ?)",
            (month, month, budget_limit))
        conn.commit()


def add_savings_goal(goal_name, target_amount, target_date):
    with sqlite3.connect('finance.db') as conn:
        c = conn.cursor()
        c.execute(
            "INSERT INTO savings_goals (goal_name, target_amount, current_amount, target_date) VALUES (?, ?, 0, ?)",
            (goal_name, target_amount, target_date))
        conn.commit()


def update_savings_goal(goal_name, amount):
    with sqlite3.connect('finance.db') as conn:
        c = conn.cursor()
        c.execute("UPDATE savings_goals SET current_amount = current_amount + ? WHERE goal_name = ?",
                  (amount, goal_name))
        conn.commit()


def get_expenses():
    with sqlite3.connect('finance.db') as conn:
        df = pd.read_sql_query("SELECT * FROM expenses", conn)
    return df


def get_income():
    with sqlite3.connect('finance.db') as conn:
        df = pd.read_sql_query("SELECT * FROM income", conn)
    return df


def get_budget(month):
    with sqlite3.connect('finance.db') as conn:
        c = conn.cursor()
        c.execute("SELECT budget_limit FROM budget WHERE month = ?", (month,))
        result = c.fetchone()
    return result[0] if result else None


def get_savings_goals():
    with sqlite3.connect('finance.db') as conn:
        df = pd.read_sql_query("SELECT * FROM savings_goals", conn)
    return df


# Enhanced Sentiment Analysis
def analyze_sentiment(text):
    if not text or not text.strip():
        return 0.0
    blob = TextBlob(text)
    return blob.sentiment.polarity


# Enhanced Spending Prediction
def predict_spending():
    df = get_expenses()
    if df.empty or 'date' not in df.columns or 'amount' not in df.columns:
        return 0.0
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    monthly_expenses = df.groupby(['year', 'month'])['amount'].sum().reset_index()
    if monthly_expenses.empty:
        return 0.0
    X = monthly_expenses[['year', 'month']]
    y = monthly_expenses['amount']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    current_date = datetime.now()
    next_month = pd.DataFrame({'year': [current_date.year], 'month': [(current_date.month % 12) + 1]})
    return float(model.predict(next_month)[0])


# Enhanced Income Prediction
def predict_income():
    df = get_income()
    if df.empty or 'date' not in df.columns or 'amount' not in df.columns:
        return 0.0
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    monthly_income = df.groupby(['year', 'month'])['amount'].sum().reset_index()
    if monthly_income.empty:
        return 0.0
    X = monthly_income[['year', 'month']]
    y = monthly_income['amount']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    current_date = datetime.now()
    next_month = pd.DataFrame({'year': [current_date.year], 'month': [(current_date.month % 12) + 1]})
    return float(model.predict(next_month)[0])


# Enhanced AI Chatbot with better responses
def get_chatbot_response(user_input):
    user_input = user_input.lower().strip()

    # Savings advice
    if any(word in user_input for word in ['save', 'saving', 'emergency fund']):
        return "💡 Build an emergency fund first (3-6 months expenses), then save 20% of income. Automate transfers to savings accounts for consistency."

    # Budget advice
    elif any(word in user_input for word in ['budget', 'spending', 'expenses']):
        return "📊 Follow the 50/30/20 rule: 50% needs, 30% wants, 20% savings. Track every expense and review monthly to identify spending patterns."

    # Investment advice
    elif 'invest' in user_input:
        return "📈 Start with low-cost index funds or ETFs. Diversify across asset classes and invest consistently over time. Consider your risk tolerance and timeline."

    # Debt management
    elif any(word in user_input for word in ['debt', 'loan', 'credit']):
        return "💳 Pay minimums on all debts, then focus extra payments on highest interest rate debt (avalanche method). Consider debt consolidation if beneficial."

    # Retirement planning
    elif any(word in user_input for word in ['retirement', '401k', 'pension']):
        return "🏖️ Start early! Contribute enough to get employer match, then maximize tax-advantaged accounts. Aim to save 10-15% of income for retirement."

    # General financial health
    elif any(word in user_input for word in ['financial health', 'money management', 'financial tips']):
        return "💪 Focus on: 1) Building emergency fund, 2) Paying off high-interest debt, 3) Budgeting effectively, 4) Investing for long-term goals, 5) Protecting with insurance."

    return "🤖 I can help with saving strategies, budgeting tips, investment guidance, debt management, and retirement planning. What specific area interests you?"


# Export enhanced data
def export_data():
    expenses = get_expenses()
    income = get_income()
    if expenses.empty and income.empty:
        return "No data to export"

    # Combine and enhance data
    expenses_enhanced = expenses.copy()
    if not expenses_enhanced.empty:
        expenses_enhanced['Type'] = 'Expense'
        expenses_enhanced['source'] = expenses_enhanced['category']

    income_enhanced = income.copy()
    if not income_enhanced.empty:
        income_enhanced['Type'] = 'Income'
        income_enhanced['category'] = income_enhanced['source']

    combined = pd.concat([expenses_enhanced, income_enhanced], ignore_index=True)
    csv_buffer = io.StringIO()
    combined.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()


# Create custom metric cards
def create_metric_card(title, value, delta=None, delta_color="normal"):
    delta_html = ""
    if delta:
        color = "#10B981" if delta_color == "normal" else "#EF4444"
        delta_html = f'<div style="color: {color}; font-size: 0.8rem; margin-top: 4px;">{delta}</div>'

    return f"""
    <div style="
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 16px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    ">
        <div style="color: #8B5CF6; font-size: 0.9rem; font-weight: 500; margin-bottom: 8px;">{title}</div>
        <div style="color: white; font-size: 2rem; font-weight: 700;">{value}</div>
        {delta_html}
    </div>
    """


# Streamlit Dashboard with Premium UI
def main():
    init_db()

    st.set_page_config(
        page_title="FinanceAI Pro",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Premium CSS styling
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        /* Global Styles */
        .main {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }

        .stApp {
            background: linear-gradient(135deg, #1e1e2e 0%, #2d1b69 50%, #11101d 100%);
            color: #ffffff;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Header Styling */
        .main-header {
            background: linear-gradient(135deg, rgba(139,92,246,0.3) 0%, rgba(59,130,246,0.3) 100%);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 24px;
            padding: 2rem;
            margin: 1rem 0 2rem 0;
            text-align: center;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }

        .main-title {
            font-size: 3.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #8B5CF6 0%, #3B82F6 50%, #06B6D4 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.5rem;
            letter-spacing: -0.02em;
        }

        .main-subtitle {
            font-size: 1.2rem;
            color: #D1D5DB;
            font-weight: 400;
            opacity: 0.9;
        }

        /* Sidebar Styling */
        .css-1d391kg {
            background: linear-gradient(180deg, rgba(30,30,46,0.95) 0%, rgba(17,16,29,0.95) 100%);
            backdrop-filter: blur(20px);
            border-right: 1px solid rgba(139,92,246,0.3);
            box-shadow: 4px 0 20px rgba(0,0,0,0.1);
        }

        /* Enhanced Button Styling */
        .stButton > button {
            background: linear-gradient(135deg, #8B5CF6 0%, #3B82F6 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            font-size: 0.9rem;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 12px rgba(139,92,246,0.3);
            width: 100%;
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(139,92,246,0.4);
            background: linear-gradient(135deg, #7C3AED 0%, #2563EB 100%);
        }

        .stButton > button:active {
            transform: translateY(0);
        }

        /* Input Field Styling */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div > div > div,
        .stTextArea > div > div > textarea {
            background: rgba(255,255,255,0.1) !important;
            border: 1px solid rgba(139,92,246,0.3) !important;
            border-radius: 12px !important;
            color: white !important;
            backdrop-filter: blur(10px) !important;
            transition: all 0.3s ease !important;
        }

        .stTextInput > div > div > input:focus,
        .stNumberInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus {
            border-color: #8B5CF6 !important;
            box-shadow: 0 0 0 3px rgba(139,92,246,0.1) !important;
        }

        /* Label Styling */
        .stTextInput > label,
        .stNumberInput > label,
        .stSelectbox > label,
        .stTextArea > label,
        .stDateInput > label {
            color: #D1D5DB !important;
            font-weight: 500 !important;
            font-size: 0.9rem !important;
        }

        /* Expander Styling */
        .streamlit-expanderHeader {
            background: linear-gradient(135deg, rgba(139,92,246,0.2) 0%, rgba(59,130,246,0.2) 100%) !important;
            border: 1px solid rgba(139,92,246,0.3) !important;
            border-radius: 12px !important;
            color: white !important;
            font-weight: 600 !important;
            margin: 0.5rem 0 !important;
        }

        .streamlit-expanderContent {
            background: rgba(30,30,46,0.5) !important;
            border-radius: 0 0 12px 12px !important;
            border: 1px solid rgba(139,92,246,0.2) !important;
            border-top: none !important;
            backdrop-filter: blur(10px) !important;
        }

        /* Chart Container */
        .chart-container {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 16px;
            padding: 1rem;
            margin: 1rem 0;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }

        /* Success/Error Messages */
        .stSuccess {
            background: rgba(16,185,129,0.1) !important;
            border: 1px solid rgba(16,185,129,0.3) !important;
            border-radius: 12px !important;
            color: #10B981 !important;
        }

        .stError {
            background: rgba(239,68,68,0.1) !important;
            border: 1px solid rgba(239,68,68,0.3) !important;
            border-radius: 12px !important;
            color: #EF4444 !important;
        }

        /* Progress Bar */
        .stProgress .st-bo {
            background: linear-gradient(135deg, #8B5CF6 0%, #3B82F6 100%) !important;
            border-radius: 8px !important;
        }

        /* Table Styling */
        .stTable {
            background: rgba(255,255,255,0.05) !important;
            border-radius: 12px !important;
            overflow: hidden !important;
        }

        /* Metric Cards */
        .metric-card {
            background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 16px;
            padding: 1.5rem;
            margin: 0.5rem 0;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }

        .metric-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 48px rgba(0,0,0,0.2);
        }

        /* Hide Streamlit Elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Animation Classes */
        .fade-in {
            animation: fadeIn 0.8s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .slide-up {
            animation: slideUp 0.6s ease-out;
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(40px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            .main-title { font-size: 2.5rem; }
            .main-header { padding: 1.5rem; }
        }
        </style>
    """, unsafe_allow_html=True)

    # Main Header
    st.markdown("""
        <div class="main-header fade-in">
            <h1 class="main-title">💰 FinanceAI Pro</h1>
            <p class="main-subtitle">Intelligent Financial Management & Analytics Platform</p>
        </div>
    """, unsafe_allow_html=True)

    # Enhanced Sidebar
    with st.sidebar:
        st.markdown("""
            <div style="text-align: center; padding: 1rem 0; border-bottom: 1px solid rgba(139,92,246,0.3); margin-bottom: 1rem;">
                <h2 style="color: #8B5CF6; margin: 0; font-weight: 700;">Control Center</h2>
                <p style="color: #9CA3AF; font-size: 0.9rem; margin: 0.5rem 0 0 0;">Manage your finances</p>
            </div>
        """, unsafe_allow_html=True)

        # Add Expense Section
        with st.expander("💸 Add Expense", expanded=False):
            st.markdown("**Record a new expense**")
            date_exp = st.date_input("📅 Date", datetime.now(), key="expense_date")
            amount_exp = st.number_input("💵 Amount ($)", min_value=0.0, format="%.2f", key="expense_amount")
            category_exp = st.selectbox("🏷️ Category",
                                        ["🛒 Groceries", "⚡ Utilities", "🎬 Entertainment", "✈️ Travel", "🏠 Housing",
                                         "🚗 Transportation", "👕 Clothing", "🏥 Healthcare", "📚 Education", "🍽️ Dining",
                                         "📱 Technology", "🔧 Other"],
                                        key="expense_category")

            if st.button("💾 Log Expense", key="add_expense"):
                add_expense(date_exp.strftime('%Y-%m-%d'), amount_exp, category_exp)
                st.success("✅ Expense logged successfully!")

        # Add Income Section
        with st.expander("💰 Add Income", expanded=False):
            st.markdown("**Record new income**")
            date_inc = st.date_input("📅 Date", datetime.now(), key="income_date")
            amount_inc = st.number_input("💵 Amount ($)", min_value=0.0, format="%.2f", key="income_amount")
            source_inc = st.selectbox("💼 Source",
                                      ["💼 Salary", "🏢 Freelance", "📈 Investment", "🎁 Gift", "💸 Bonus", "🏠 Rental",
                                       "💰 Side Hustle", "🔧 Other"],
                                      key="income_source")

            if st.button("💾 Log Income", key="add_income"):
                add_income(date_inc.strftime('%Y-%m-%d'), amount_inc, source_inc)
                st.success("✅ Income logged successfully!")

        # Budget Section
        with st.expander("🎯 Set Budget", expanded=False):
            st.markdown("**Monthly budget planning**")
            month = st.text_input("📆 Month (YYYY-MM)", datetime.now().strftime("%Y-%m"), key="budget_month")
            budget_limit = st.number_input("🏦 Budget Limit ($)", min_value=0.0, format="%.2f", key="budget_limit")

            if st.button("🎯 Set Budget", key="set_budget"):
                set_budget(month, budget_limit)
                st.success("✅ Budget set successfully!")

        # Savings Goals Section
        with st.expander("🎯 Savings Goals", expanded=False):
            st.markdown("**Financial goal tracking**")

            # Add new goal
            st.markdown("**Create New Goal**")
            goal_name = st.text_input("🎯 Goal Name", placeholder="e.g., Emergency Fund", key="goal_name")
            target_amount = st.number_input("💰 Target Amount ($)", min_value=0.0, format="%.2f", key="target_amount")
            target_date = st.date_input("📅 Target Date", datetime.now() + timedelta(days=365), key="target_date")

            if st.button("🎯 Create Goal", key="add_goal"):
                if goal_name and target_amount > 0:
                    add_savings_goal(goal_name, target_amount, target_date.strftime('%Y-%m-%d'))
                    st.success("✅ Goal created successfully!")
                else:
                    st.error("Please fill in all fields")

            # Update existing goal
            st.markdown("**Add to Existing Goal**")
            goals_df = get_savings_goals()
            if not goals_df.empty:
                goal_to_update = st.selectbox("Select Goal", goals_df['goal_name'].tolist(), key="select_goal")
                amount_to_add = st.number_input("💵 Amount to Add ($)", min_value=0.0, format="%.2f",
                                                key="update_amount")

                if st.button("➕ Add to Goal", key="add_to_goal"):
                    if amount_to_add > 0:
                        update_savings_goal(goal_to_update, amount_to_add)
                        st.success(f"✅ Added ${amount_to_add:.2f} to {goal_to_update}!")

        # Sentiment Analysis Section
        st.markdown("""
            <div style="text-align: center; padding: 1rem 0; border-top: 1px solid rgba(139,92,246,0.3); margin-top: 1.5rem;">
                <h3 style="color: #8B5CF6; margin: 0 0 0.5rem 0; font-weight: 600;">💭 Mood Tracker</h3>
                <p style="color: #9CA3AF; font-size: 0.85rem; margin: 0;">How are you feeling about your finances?</p>
            </div>
        """, unsafe_allow_html=True)

        user_text = st.text_area("✍️ Share your thoughts...",
                                 placeholder="e.g., Feeling stressed about my spending this month...",
                                 key="sentiment_text", height=100)

        if st.button("🔍 Analyze Sentiment", key="analyze_sentiment"):
            if user_text.strip():
                sentiment = analyze_sentiment(user_text)
                add_sentiment(datetime.now().strftime('%Y-%m-%d'), sentiment, "user")

                # Enhanced sentiment feedback
                if sentiment < -0.5:
                    st.error(
                        f"😰 Very Negative Sentiment ({sentiment:.2f}) - Consider speaking with a financial advisor")
                elif sentiment < -0.1:
                    st.warning(f"😕 Negative Sentiment ({sentiment:.2f}) - Focus on small wins and budgeting")
                elif sentiment < 0.1:
                    st.info(f"😐 Neutral Sentiment ({sentiment:.2f}) - Stay consistent with your financial habits")
                elif sentiment < 0.5:
                    st.success(f"😊 Positive Sentiment ({sentiment:.2f}) - Great mindset for financial growth!")
                else:
                    st.success(f"🎉 Very Positive Sentiment ({sentiment:.2f}) - Excellent financial confidence!")

    # Main Dashboard Content
    st.markdown('<div class="slide-up">', unsafe_allow_html=True)

    # Key Metrics Row
    expenses_df = get_expenses()
    income_df = get_income()
    current_month = datetime.now().strftime("%Y-%m")

    # Calculate key metrics
    if not expenses_df.empty:
        monthly_expenses = expenses_df[pd.to_datetime(expenses_df['date']).dt.strftime("%Y-%m") == current_month][
            'amount'].sum()
        total_expenses = expenses_df['amount'].sum()
    else:
        monthly_expenses = 0
        total_expenses = 0

    if not income_df.empty:
        monthly_income = income_df[pd.to_datetime(income_df['date']).dt.strftime("%Y-%m") == current_month][
            'amount'].sum()
        total_income = income_df['amount'].sum()
    else:
        monthly_income = 0
        total_income = 0

    net_worth = total_income - total_expenses
    monthly_savings = monthly_income - monthly_expenses

    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(create_metric_card(
            "💰 Monthly Income",
            f"${monthly_income:,.2f}",
            f"+${monthly_income - (monthly_income * 0.9):,.2f}" if monthly_income > 0 else None
        ), unsafe_allow_html=True)

    with col2:
        st.markdown(create_metric_card(
            "💸 Monthly Expenses",
            f"${monthly_expenses:,.2f}",
            f"+${monthly_expenses - (monthly_expenses * 0.9):,.2f}" if monthly_expenses > 0 else None,
            "inverse" if monthly_expenses > monthly_income else "normal"
        ), unsafe_allow_html=True)

    with col3:
        st.markdown(create_metric_card(
            "💎 Monthly Savings",
            f"${monthly_savings:,.2f}",
            "🎯 Great job!" if monthly_savings > 0 else "⚠️ Overspending"
        ), unsafe_allow_html=True)

    with col4:
        st.markdown(create_metric_card(
            "🏦 Net Worth",
            f"${net_worth:,.2f}",
            "📈 Growing" if net_worth > 0 else "📉 Deficit"
        ), unsafe_allow_html=True)

    # Charts Section
    st.markdown("""
        <div style="margin: 2rem 0;">
            <h2 style="color: #8B5CF6; text-align: center; margin-bottom: 2rem; font-weight: 700; font-size: 2rem;">
                📊 Financial Analytics Dashboard
            </h2>
        </div>
    """, unsafe_allow_html=True)

    # Charts Row 1
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        if not expenses_df.empty:
            st.markdown("### 🍕 Expense Distribution")
            # Clean category names for display
            expenses_display = expenses_df.copy()
            expenses_display['category'] = expenses_display['category'].str.replace(r'[🛒⚡🎬✈️🏠🚗👕🏥📚🍽️📱🔧]', '',
                                                                                    regex=True).str.strip()

            fig_pie = px.pie(
                expenses_display,
                names="category",
                values="amount",
                title="",
                color_discrete_sequence=px.colors.qualitative.Set3,
                hole=0.4
            )
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', size=12),
                showlegend=True,
                legend=dict(orientation="v", x=1.05, y=0.5)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("💡 Add some expenses to see the distribution chart")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        if not income_df.empty:
            st.markdown("### 💰 Income Sources")
            income_display = income_df.copy()
            income_display['source'] = income_display['source'].str.replace(r'[💼🏢📈🎁💸🏠💰🔧]', '', regex=True).str.strip()

            fig_income = px.bar(
                income_display.groupby('source')['amount'].sum().reset_index(),
                x='source',
                y='amount',
                title="",
                color='amount',
                color_continuous_scale='Viridis'
            )
            fig_income.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', size=12),
                xaxis=dict(showgrid=False, color='white'),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', color='white'),
                showlegend=False
            )
            st.plotly_chart(fig_income, use_container_width=True)
        else:
            st.info("💡 Add some income to see the sources chart")
        st.markdown('</div>', unsafe_allow_html=True)

    # Charts Row 2
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        if not expenses_df.empty:
            st.markdown("### 📈 Spending Trends")
            expenses_time = expenses_df.copy()
            expenses_time['date'] = pd.to_datetime(expenses_time['date'])
            expenses_time['month'] = expenses_time['date'].dt.to_period('M')
            monthly_spending = expenses_time.groupby('month')['amount'].sum().reset_index()
            monthly_spending['month'] = monthly_spending['month'].astype(str)

            fig_trend = px.line(
                monthly_spending,
                x='month',
                y='amount',
                title="",
                markers=True,
                line_shape='spline'
            )
            fig_trend.update_traces(
                line=dict(color='#8B5CF6', width=3),
                marker=dict(size=8, color='#3B82F6')
            )
            fig_trend.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', size=12),
                xaxis=dict(showgrid=False, color='white'),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', color='white')
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("💡 Add expenses over time to see spending trends")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("### 🎯 Savings Goals Progress")
        goals_df = get_savings_goals()
        if not goals_df.empty:
            for _, goal in goals_df.iterrows():
                progress = min(goal['current_amount'] / goal['target_amount'], 1.0) * 100
                days_left = (pd.to_datetime(goal['target_date']) - datetime.now()).days

                # Progress bar with custom styling
                st.markdown(f"""
                    <div style="margin: 1rem 0; padding: 1rem; background: rgba(255,255,255,0.05); border-radius: 12px; border: 1px solid rgba(139,92,246,0.2);">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                            <span style="font-weight: 600; color: #8B5CF6;">{goal['goal_name']}</span>
                            <span style="color: #D1D5DB;">${goal['current_amount']:,.0f} / ${goal['target_amount']:,.0f}</span>
                        </div>
                        <div style="background: rgba(255,255,255,0.1); border-radius: 8px; height: 8px; overflow: hidden;">
                            <div style="background: linear-gradient(135deg, #8B5CF6 0%, #3B82F6 100%); height: 100%; width: {progress}%; transition: width 0.3s ease;"></div>
                        </div>
                        <div style="margin-top: 0.5rem; font-size: 0.85rem; color: #9CA3AF;">
                            {progress:.1f}% complete • {days_left} days remaining
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("💡 Set up savings goals to track your progress")
        st.markdown('</div>', unsafe_allow_html=True)

    # Budget Analysis Section
    st.markdown("""
        <div style="margin: 2rem 0;">
            <h2 style="color: #8B5CF6; text-align: center; margin-bottom: 2rem; font-weight: 700; font-size: 2rem;">
                🎯 Budget Analysis & Insights
            </h2>
        </div>
    """, unsafe_allow_html=True)

    budget_limit = get_budget(current_month)
    if budget_limit is not None:
        budget_used = (monthly_expenses / budget_limit) * 100 if budget_limit > 0 else 0
        remaining_budget = budget_limit - monthly_expenses

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(create_metric_card(
                "🏦 Monthly Budget",
                f"${budget_limit:,.2f}",
                "Set for this month"
            ), unsafe_allow_html=True)

        with col2:
            color = "normal" if budget_used <= 80 else "inverse"
            st.markdown(create_metric_card(
                "📊 Budget Used",
                f"{budget_used:.1f}%",
                "🟢 On track" if budget_used <= 80 else "🔴 Over budget" if budget_used > 100 else "🟡 Close to limit"
            ), unsafe_allow_html=True)

        with col3:
            st.markdown(create_metric_card(
                "💰 Remaining Budget",
                f"${remaining_budget:,.2f}",
                "Available to spend" if remaining_budget > 0 else "Overspent!"
            ), unsafe_allow_html=True)
    else:
        st.info("💡 Set a monthly budget to see detailed analysis and insights")

    # AI Predictions Section
    st.markdown("""
        <div style="margin: 2rem 0;">
            <h2 style="color: #8B5CF6; text-align: center; margin-bottom: 2rem; font-weight: 700; font-size: 2rem;">
                🤖 AI Predictions & Recommendations
            </h2>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        predicted_spending = predict_spending()
        predicted_income = predict_income()
        predicted_savings = predicted_income - predicted_spending

        st.markdown(create_metric_card(
            "🔮 Predicted Spending (Next Month)",
            f"${predicted_spending:,.2f}",
            f"Based on {len(expenses_df)} transactions" if not expenses_df.empty else "Add more data for accuracy"
        ), unsafe_allow_html=True)

        st.markdown(create_metric_card(
            "💰 Predicted Income (Next Month)",
            f"${predicted_income:,.2f}",
            f"Based on {len(income_df)} records" if not income_df.empty else "Add more data for accuracy"
        ), unsafe_allow_html=True)

    with col2:
        st.markdown(create_metric_card(
            "💎 Predicted Savings",
            f"${predicted_savings:,.2f}",
            "🎯 Excellent!" if predicted_savings > 0 else "⚠️ Consider reducing expenses"
        ), unsafe_allow_html=True)

        # AI Recommendations
        st.markdown("""
            <div style="
                background: linear-gradient(135deg, rgba(139,92,246,0.1) 0%, rgba(59,130,246,0.1) 100%);
                border: 1px solid rgba(139,92,246,0.3);
                border-radius: 16px;
                padding: 1.5rem;
                margin: 1rem 0;
            ">
                <h4 style="color: #8B5CF6; margin: 0 0 1rem 0;">💡 AI Recommendations</h4>
        """, unsafe_allow_html=True)

        # Generate recommendations based on data
        recommendations = []
        if monthly_expenses > monthly_income:
            recommendations.append("🔴 Reduce expenses by focusing on your largest spending categories")
        if monthly_savings < monthly_income * 0.2:
            recommendations.append("🟡 Try to save at least 20% of your income")
        if budget_limit and monthly_expenses > budget_limit * 0.8:
            recommendations.append("🟡 You're approaching your budget limit - monitor spending carefully")
        if not recommendations:
            recommendations.append("🟢 Your financial habits look healthy - keep it up!")

        for rec in recommendations:
            st.markdown(f"<p style='margin: 0.5rem 0; color: #D1D5DB;'>• {rec}</p>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # Interactive AI Assistant
    st.markdown("""
        <div style="margin: 2rem 0;">
            <h2 style="color: #8B5CF6; text-align: center; margin-bottom: 2rem; font-weight: 700; font-size: 2rem;">
                🤖 AI Financial Assistant
            </h2>
        </div>
    """, unsafe_allow_html=True)

    user_question = st.text_input(
        "💬 Ask your AI assistant anything about finance:",
        placeholder="e.g., How can I save more money? What's the best way to pay off debt?",
        key="ai_question"
    )

    if user_question:
        with st.container():
            response = get_chatbot_response(user_question)
            st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, rgba(59,130,246,0.1) 0%, rgba(16,185,129,0.1) 100%);
                    border: 1px solid rgba(59,130,246,0.3);
                    border-radius: 16px;
                    padding: 1.5rem;
                    margin: 1rem 0;
                ">
                    <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                        <span style="font-size: 1.5rem; margin-right: 0.5rem;">🤖</span>
                        <strong style="color: #3B82F6;">AI Assistant</strong>
                    </div>
                    <p style="margin: 0; color: #D1D5DB; line-height: 1.6;">{response}</p>
                </div>
            """, unsafe_allow_html=True)

    # Recent Transactions
    st.markdown("""
        <div style="margin: 2rem 0;">
            <h2 style="color: #8B5CF6; text-align: center; margin-bottom: 2rem; font-weight: 700; font-size: 2rem;">
                📋 Recent Transactions
            </h2>
        </div>
    """, unsafe_allow_html=True)

    # Combine and display recent transactions
    recent_transactions = []
    if not expenses_df.empty:
        recent_expenses = expenses_df.tail(5).copy()
        recent_expenses['Type'] = '💸 Expense'
        recent_expenses['Description'] = recent_expenses['category']
        recent_transactions.append(recent_expenses[['date', 'amount', 'Description', 'Type']])

    if not income_df.empty:
        recent_income = income_df.tail(5).copy()
        recent_income['Type'] = '💰 Income'
        recent_income['Description'] = recent_income['source']
        recent_transactions.append(recent_income[['date', 'amount', 'Description', 'Type']])

    if recent_transactions:
        all_recent = pd.concat(recent_transactions, ignore_index=True)
        all_recent['date'] = pd.to_datetime(all_recent['date'])
        all_recent = all_recent.sort_values('date', ascending=False).head(10)

        # Display in a styled format
        for _, transaction in all_recent.iterrows():
            amount_color = "#10B981" if "Income" in transaction['Type'] else "#EF4444"
            amount_prefix = "+" if "Income" in transaction['Type'] else "-"

            st.markdown(f"""
                <div style="
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    background: rgba(255,255,255,0.05);
                    border: 1px solid rgba(255,255,255,0.1);
                    border-radius: 12px;
                    padding: 1rem;
                    margin: 0.5rem 0;
                ">
                    <div style="display: flex; align-items: center;">
                        <span style="margin-right: 1rem;">{transaction['Type']}</span>
                        <div>
                            <div style="font-weight: 600; color: white;">{transaction['Description']}</div>
                            <div style="font-size: 0.85rem; color: #9CA3AF;">{transaction['date'].strftime('%B %d, %Y')}</div>
                        </div>
                    </div>
                    <div style="font-weight: 700; font-size: 1.1rem; color: {amount_color};">
                        {amount_prefix}${transaction['amount']:,.2f}
                    </div>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("💡 Start by adding some income and expenses to see your transaction history")

    # Export Section
    st.markdown("""
        <div style="margin: 2rem 0;">
            <h2 style="color: #8B5CF6; text-align: center; margin-bottom: 2rem; font-weight: 700; font-size: 2rem;">
                📥 Export Your Data
            </h2>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Download Financial Report (CSV)", key="download_csv"):
            csv_data = export_data()
            if csv_data != "No data to export":
                st.download_button(
                    label="💾 Download CSV File",
                    data=csv_data,
                    file_name=f"financial_report_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    key="csv_download"
                )
                st.success("✅ Report ready for download!")
            else:
                st.warning("⚠️ No data available to export")

    with col2:
        st.info("💡 Your financial data will be exported in CSV format, perfect for Excel or Google Sheets analysis")

    st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("""
        <div style="
            margin-top: 4rem;
            padding: 2rem;
            text-align: center;
            border-top: 1px solid rgba(139,92,246,0.2);
            background: linear-gradient(135deg, rgba(139,92,246,0.1) 0%, rgba(59,130,246,0.1) 100%);
            border-radius: 16px;
        ">
            <p style="margin: 0; color: #9CA3AF; font-size: 0.9rem;">
                💰 <strong>FinanceAI Pro</strong> - Your Intelligent Financial Management Platform
            </p>
            <p style="margin: 0.5rem 0 0 0; color: #6B7280; font-size: 0.8rem;">
                Built with AI-powered insights • Secure • Privacy-first • Professional-grade analytics
            </p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()