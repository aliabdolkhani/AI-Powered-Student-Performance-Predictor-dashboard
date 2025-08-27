# app.py
# Streamlit Dashboard for Student Performance Feedback with Login, GPT Feedback, ARIMA Forecasting, and PDF Export

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import io
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.dates as mdates

# === CONFIG ===
client = OpenAI(api_key=st.secrets["general"]["openai_api_key"])

# Load data
df = pd.read_csv("student_performance.csv")
df['attempt_date'] = pd.to_datetime(df['attempt_date'])
df['year_month'] = df['attempt_date'].dt.to_period('M')

# Add student names
names = [
    "Alex", "Bella", "Charlie", "Dina", "Eli", "Fatima", "George", "Hana",
    "Isaac", "Julia", "Kevin", "Lana", "Mike", "Nina", "Omar", "Paula",
    "Quinn", "Rosa", "Sam", "Tina"
]
student_ids = df['student_id'].unique()
name_map = dict(zip(student_ids,
                    pd.Series(names).sample(n=len(student_ids), replace=True, random_state=42).values))
df['student_name'] = df['student_id'].map(name_map)

# ===================
# Session State Setup
# ===================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "student_id" not in st.session_state:
    st.session_state.student_id = None

# ===================
# Login Page
# ===================
def login_page():
    st.title("🔐 Student Login")

    user_input = st.text_input("Student ID")
    pass_input = st.text_input("Password", type="password")
    login_button = st.button("Login")

    if login_button:
        if user_input in df['student_id'].unique() and pass_input == user_input:
            st.session_state.logged_in = True
            st.session_state.student_id = user_input
            st.success("Login successful! Redirecting...")
            st.rerun()
        else:
            st.error("Invalid credentials. Please try again.")

# ===================
# PDF Generator
# ===================
def generate_pdf(student_name, accuracy_by_subject, trend_fig, incorrect_df, feedback, forecast, student_df, arima_fig=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph(f"Student Performance Report - {student_name}", styles["Title"]))
    story.append(Spacer(1, 12))

    # Performance Table
    story.append(Paragraph("Performance by Subject", styles["Heading2"]))
    table_data = [accuracy_by_subject.reset_index().columns.tolist()] + accuracy_by_subject.reset_index().values.tolist()
    table = Table(table_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND',(0,1),(-1,-1),colors.beige),
    ]))
    story.append(table)
    story.append(Spacer(1, 12))

    # Trend Chart (monthly accuracy, white background for PDF readability)
    story.append(Paragraph("Monthly Accuracy Trend", styles["Heading2"]))
    img_buf = io.BytesIO()
    trend_fig.savefig(img_buf, format="png")
    img_buf.seek(0)
    story.append(Image(img_buf, width=400, height=200))
    story.append(Spacer(1, 12))

    # Subject-level last 2 months weekly comparison (white in PDF)
    story.append(Paragraph("Subject-Level Weekly Comparison (Last 2 Months)", styles["Heading2"]))
    last_two_months = student_df['year_month'].dropna().unique()
    last_two_months = sorted(last_two_months)[-2:]

    if len(last_two_months) >= 2:
        for subj in student_df['subject'].unique():
            story.append(Paragraph(f"{subj}", styles["Heading3"]))

            subj_data = student_df[student_df['subject'] == subj].copy()
            subj_data['week'] = subj_data['attempt_date'].dt.to_period('W')
            subj_recent = subj_data[subj_data['year_month'].isin(last_two_months)]
            grouped = subj_recent.groupby(['year_month','week'])['is_correct'].mean().reset_index()

            fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
            for i, month in enumerate(last_two_months):
                month_data = grouped[grouped['year_month'] == month]
                axes[i].plot(month_data['week'].astype(str),
                             month_data['is_correct'] * 100,
                             marker='o')
                axes[i].set_title(str(month))
                axes[i].set_ylim(0, 100)
                axes[i].set_xticklabels(month_data['week'].astype(str), rotation=45)

            fig.suptitle(f"Weekly Accuracy for {subj} (Last 2 Months)")
            img_buf = io.BytesIO()
            fig.savefig(img_buf, format="png", bbox_inches="tight")
            plt.close(fig)
            img_buf.seek(0)
            story.append(Image(img_buf, width=400, height=180))
            story.append(Spacer(1, 12))

    # Incorrect Questions
    story.append(Paragraph("Incorrect Questions", styles["Heading2"]))
    if incorrect_df.empty:
        story.append(Paragraph("Great job! You answered all questions correctly.", styles["Normal"]))
    else:
        for _, row in incorrect_df.iterrows():
            story.append(Paragraph(f"{row['attempt_date'].date()} - {row['subject']}: {row['question_text']} "
                                   f"(Answered: {row['student_answer']}, Correct: {row['correct_answer']})",
                                   styles["Normal"]))
            story.append(Spacer(1, 6))

    # GPT Feedback
    story.append(Spacer(1, 12))
    story.append(Paragraph("Personalized Feedback", styles["Heading2"]))
    story.append(Paragraph(feedback, styles["Normal"]))

    # ARIMA Forecast Chart
    if arima_fig:
        story.append(Spacer(1, 12))
        story.append(Paragraph("ARIMA Forecast Chart", styles["Heading2"]))
        img_buf = io.BytesIO()
        arima_fig.savefig(img_buf, format="png", bbox_inches="tight")
        plt.close(arima_fig)
        img_buf.seek(0)
        story.append(Image(img_buf, width=400, height=200))

    # GPT Forecast (hybrid ARIMA+GPT text)
    story.append(Spacer(1, 12))
    story.append(Paragraph("Forecast", styles["Heading2"]))
    story.append(Paragraph(forecast, styles["Normal"]))

    doc.build(story)
    pdf_value = buffer.getvalue()
    buffer.close()
    return pdf_value

# ===================
# Dashboard Page
# ===================
def dashboard_page():
    student_id = st.session_state.student_id
    student_df = df[df['student_id'] == student_id]
    student_name = student_df['student_name'].iloc[0]

    st.header(f"📊 Welcome, {student_name}!")

    # Accuracy by subject
    accuracy_by_subject = (student_df.groupby('subject')['is_correct']
                           .agg(['count', 'sum'])
                           .rename(columns={'count': 'Total', 'sum': 'Correct'}))
    accuracy_by_subject['Accuracy (%)'] = 100 * accuracy_by_subject['Correct'] / accuracy_by_subject['Total']

    st.subheader("Performance by Subject")
    st.dataframe(accuracy_by_subject.style.format("{:.1f}"))

    # Multi-line Trend (with black background in app)
    trend = (student_df.groupby(['year_month', 'subject'])['is_correct']
             .mean()
             .reset_index()
             .rename(columns={'is_correct': 'Accuracy'}))

    st.subheader("📈 Monthly Accuracy Trend (Comparison)")
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 4))
    for subj in trend['subject'].unique():
        subj_data = trend[trend['subject'] == subj]
        ax.plot(subj_data['year_month'].astype(str),
                subj_data['Accuracy'] * 100,
                marker='o',
                label=subj)
    ax.set_title("Accuracy Over Time by Subject", color="white")
    ax.set_xticklabels(subj_data['year_month'].astype(str), rotation=45, color="white")
    ax.set_ylim(0, 100)
    ax.tick_params(axis='y', colors="white")
    ax.tick_params(axis='x', colors="white")
    ax.legend()
    st.pyplot(fig)
    plt.style.use("default")

    # Subject-level last 2 months weekly comparison (black background in app)
    st.subheader("📊 Subject-Level Comparison (Last 2 Months)")
    last_two_months = student_df['year_month'].dropna().unique()
    last_two_months = sorted(last_two_months)[-2:]

    if len(last_two_months) < 2:
        st.info("Not enough data for side-by-side monthly comparison.")
    else:
        for subj in student_df['subject'].unique():
            st.markdown(f"#### {subj}")

            subj_data = student_df[student_df['subject'] == subj].copy()
            subj_data['week'] = subj_data['attempt_date'].dt.to_period('W')
            subj_recent = subj_data[subj_data['year_month'].isin(last_two_months)]
            grouped = subj_recent.groupby(['year_month','week'])['is_correct'].mean().reset_index()

            plt.style.use("dark_background")
            fig_subj, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
            for i, month in enumerate(last_two_months):
                month_data = grouped[grouped['year_month'] == month]
                axes[i].plot(month_data['week'].astype(str),
                             month_data['is_correct'] * 100,
                             marker='o')
                axes[i].set_title(str(month), color="white")
                axes[i].set_ylim(0, 100)
                axes[i].set_xticklabels(month_data['week'].astype(str), rotation=45, color="white")
                axes[i].tick_params(axis='y', colors="white")
                axes[i].tick_params(axis='x', colors="white")
            fig_subj.suptitle(f"Weekly Accuracy for {subj} (Last 2 Months)", color="white")
            st.pyplot(fig_subj)
            plt.style.use("default")

    # Show Incorrect Questions
    st.subheader("❌ Incorrect Questions")
    incorrect_df = student_df[student_df['is_correct'] == 0]
    if incorrect_df.empty:
        st.success("Great job! You answered all questions correctly.")
    else:
        st.dataframe(
            incorrect_df[['attempt_date', 'subject', 'question_text', 'student_answer', 'correct_answer']]
            .sort_values(by="attempt_date")
        )

    # GPT Feedback
    st.subheader("📝 Personalized Feedback")
    feedback = "No feedback generated."
    if not incorrect_df.empty:
        feedback_prompt = """
You are an educational assistant. Based on the following questions the student got wrong, generate personalized and constructive feedback.
Use clear, friendly language. Mention what topics need improvement and suggest 1–2 actions.
Return your response in a short paragraph.

Incorrect Questions:
""" + "\n".join(
            f"- {row['subject']}: {row['question_text']} (Answered: {row['student_answer']}, Correct: {row['correct_answer']})"
            for _, row in incorrect_df.iterrows()
        )

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful educational assistant."},
                    {"role": "user", "content": feedback_prompt}
                ]
            )
            feedback = response.choices[0].message.content.strip()
            st.markdown(feedback)
        except Exception as e:
            st.error(f"Error calling OpenAI API: {e}")
    else:
        st.success("No incorrect answers, so no feedback is needed!")

    # ARIMA Forecasting + GPT Hybrid
    st.subheader("📅 Forecast: Next Week's Accuracy")
    weekly_acc = student_df.groupby(student_df['attempt_date'].dt.to_period('W'))['is_correct'].mean()
    weekly_acc.index = weekly_acc.index.to_timestamp()

    forecast_text = "Not enough data to forecast."
    arima_value = None
    arima_fig = None

    if len(weekly_acc) >= 3:
        try:
            model = ARIMA(weekly_acc, order=(1,1,1))
            model_fit = model.fit()
            arima_value = model_fit.forecast(steps=1).values[0] * 100
            st.info(f"ARIMA Forecast (statistical model): {arima_value:.2f}%")

            # Plot observed vs forecast
            future_dates = pd.date_range(weekly_acc.index[-1] + pd.offsets.Week(), periods=1, freq="W")
            plt.style.use("dark_background")
            arima_fig, ax = plt.subplots(figsize=(8,4))
            ax.plot(weekly_acc.index, weekly_acc*100, label="Observed")
            ax.plot(future_dates, [arima_value], marker="o", label="Forecast")
            ax.set_title(f"Weekly Accuracy Forecast - {student_name}", color="white")
            ax.set_ylabel("Accuracy")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
            ax.tick_params(axis='x', colors="white", rotation=45)
            ax.tick_params(axis='y', colors="white")
            ax.legend()
            st.pyplot(arima_fig)
            plt.style.use("default")

            # Recompute monthly averages for GPT context
            trend_avg = (student_df.groupby('year_month')['is_correct'].mean().reset_index())

            # Hybrid GPT interpretation
            forecast_prompt = f"""
Based on statistical forecasting and the student’s past accuracy trends, 
please provide a short, encouraging forecast for {student_name}'s performance next week. 
Focus on motivation and a positive outlook. 
Do not mention model names or percentages, and do not repeat areas of improvement (already covered in feedback).
"""
            forecast_resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an educational assistant who combines statistical forecasts with qualitative feedback."},
                    {"role": "user", "content": forecast_prompt}
                ]
            )
            forecast_text = forecast_resp.choices[0].message.content.strip()
            st.markdown(forecast_text)

        except Exception as e:
            st.error(f"ARIMA forecast error: {e}")
    else:
        st.info("Not enough data for ARIMA forecasting.")

    # PDF Report Button
    st.subheader("📥 Download Report")
    pdf_bytes = generate_pdf(student_name, accuracy_by_subject, fig, incorrect_df, feedback, forecast_text, student_df, arima_fig)
    st.download_button(
        label="Download PDF",
        data=pdf_bytes,
        file_name=f"{student_name}_performance_report.pdf",
        mime="application/pdf"
    )

    # Logout button
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.student_id = None
        st.rerun()


# ===================
# App Navigation
# ===================
if not st.session_state.logged_in:
    login_page()
else:
    dashboard_page()
