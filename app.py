import streamlit as st
import numpy as np
import random

st.set_page_config(page_title="Smart RL Notifications", layout="centered")

# -----------------------------
# STATES & ACTIONS
# -----------------------------

states = ["Morning", "Afternoon", "Evening"]
actions = ["News", "Social", "Market", "Entertainment", "No Notification"]

notifications = {
    "Morning": {
        "News": ["🌅 Top headlines to start your day.", "☀ Weather looks great today!"],
        "Social": ["👥 You have new notifications.", "📸 Someone tagged you in a photo."],
        "Market": ["📈 Market opening update.", "💹 Stocks expected to rise."],
        "Entertainment": ["🎵 Trending morning playlist ready.", "🔮 Your daily horoscope is here."]
    },
    "Afternoon": {
        "News": ["📰 Midday news briefing.", "⚽ Live sports update."],
        "Social": ["💬 You have unread messages.", "🔥 Your post is trending."],
        "Market": ["📊 Midday market analysis.", "💰 Investment alert triggered."],
        "Entertainment": ["📺 Trending videos for lunch break.", "🎬 Recommended short clips."]
    },
    "Evening": {
        "News": ["🌙 Evening headline summary.", "📢 Top stories you missed."],
        "Social": ["❤️ Someone liked your post.", "👀 New profile view."],
        "Market": ["📉 Market closing report.", "💵 Daily portfolio summary."],
        "Entertainment": ["🍿 New episode released!", "🎥 Recommended movie tonight."]
    }
}

# -----------------------------
# SESSION STATE INIT
# -----------------------------

if "q_table" not in st.session_state:
    st.session_state.q_table = np.zeros((len(states), len(actions)))

if "epsilon" not in st.session_state:
    st.session_state.epsilon = 1.0

if "current_notification" not in st.session_state:
    st.session_state.current_notification = None

if "state_index" not in st.session_state:
    st.session_state.state_index = None

if "action_index" not in st.session_state:
    st.session_state.action_index = None

# -----------------------------
# Q-LEARNING UPDATE
# -----------------------------

def update_q(reward):

    state_index = st.session_state.state_index
    action_index = st.session_state.action_index

    learning_rate = 0.1
    discount_factor = 0.9

    old_value = st.session_state.q_table[state_index, action_index]
    next_max = np.max(st.session_state.q_table[state_index])

    new_value = old_value + learning_rate * (
        reward + discount_factor * next_max - old_value
    )

    st.session_state.q_table[state_index, action_index] = new_value

    # Decay exploration
    if st.session_state.epsilon > 0.05:
        st.session_state.epsilon *= 0.98


# -----------------------------
# UI START
# -----------------------------

st.title("🤖 Smart Notification System")
st.caption("This system learns what you prefer at different times of the day.")

st.subheader("Choose Time of Day")

col1, col2, col3 = st.columns(3)

if col1.button("🌅 Morning"):
    st.session_state.state_index = 0
    st.session_state.current_notification = None

if col2.button("☀ Afternoon"):
    st.session_state.state_index = 1
    st.session_state.current_notification = None

if col3.button("🌙 Evening"):
    st.session_state.state_index = 2
    st.session_state.current_notification = None


# -----------------------------
# GENERATE NOTIFICATION
# -----------------------------

if st.session_state.state_index is not None:

    state_index = st.session_state.state_index
    current_state = states[state_index]

    if st.session_state.current_notification is None:

        # Epsilon-greedy action selection
        if random.uniform(0, 1) < st.session_state.epsilon:
            action_index = random.randint(0, len(actions) - 1)
        else:
            action_index = np.argmax(st.session_state.q_table[state_index])

        st.session_state.action_index = action_index
        chosen_action = actions[action_index]

        if chosen_action == "No Notification":
            st.session_state.current_notification = "🔕 No notification sent."
        else:
            message = random.choice(notifications[current_state][chosen_action])
            st.session_state.current_notification = message

    # Show Notification
    st.markdown("### 📲 Notification")
    st.info(st.session_state.current_notification)

    # -----------------------------
    # FEEDBACK SECTION (WORKS FOR ALL CASES)
    # -----------------------------

    st.markdown("### 🧠 Your Feedback")

    col1, col2 = st.columns(2)

    if col1.button("👍 Good Choice"):

        if st.session_state.current_notification == "🔕 No notification sent.":
            update_q(5)      # silence rewarded
        else:
            update_q(10)     # notification rewarded

        st.success("Model Learned from Positive Feedback!")
        st.session_state.current_notification = None

    if col2.button("👎 Bad Choice"):

        if st.session_state.current_notification == "🔕 No notification sent.":
            update_q(-2)     # silence punished
        else:
            update_q(-5)     # notification punished

        st.error("Model Learned from Negative Feedback!")
        st.session_state.current_notification = None


# -----------------------------
# RESET BUTTON
# -----------------------------

st.markdown("---")

if st.button("🔄 Reset Learning"):
    st.session_state.q_table = np.zeros((len(states), len(actions)))
    st.session_state.epsilon = 1.0
    st.session_state.current_notification = None
    st.success("Model Reset!")


# -----------------------------
# DEBUG PANEL
# -----------------------------

with st.expander("📊 View Learning Data"):
    st.write("Q-Table")
    st.dataframe(st.session_state.q_table)
    st.write("Exploration (epsilon):", round(st.session_state.epsilon, 3))
