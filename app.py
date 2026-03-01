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
        "News": [
            "Good Morning! Here are today’s top headlines.",
            "Weather Update: Expect light showers today."
        ],
        "Social": [
            "You have 5 new notifications.",
            "Your friend tagged you in a post."
        ],
        "Market": [
            "Stock Market Opening Update.",
            "Nifty expected to rise today."
        ],
        "Entertainment": [
            "Start your day with trending songs.",
            "Daily horoscope is ready."
        ]
    },

    "Afternoon": {
        "News": [
            "Midday News Briefing.",
            "Live updates on current events."
        ],
        "Social": [
            "Someone reacted to your story.",
            "You have unread messages."
        ],
        "Market": [
            "Midday Market Update.",
            "Stock Alert: Price fluctuation detected."
        ],
        "Entertainment": [
            "Trending videos you may like.",
            "Lunch break watch recommendations."
        ]
    },

    "Evening": {
        "News": [
            "Evening Headlines Summary.",
            "Top stories you missed today."
        ],
        "Social": [
            "Your post is getting attention.",
            "New comments on your photo."
        ],
        "Market": [
            "Market Closing Report.",
            "Daily profit/loss summary."
        ],
        "Entertainment": [
            "Your favorite show just dropped a new episode.",
            "Recommended movies for tonight."
        ]
    }
}

# -----------------------------
# SESSION STATE INITIALIZATION
# -----------------------------

if "q_table" not in st.session_state:
    st.session_state.q_table = np.zeros((len(states), len(actions)))

if "epsilon" not in st.session_state:
    st.session_state.epsilon = 1.0

if "last_action" not in st.session_state:
    st.session_state.last_action = None

if "last_state" not in st.session_state:
    st.session_state.last_state = None


# -----------------------------
# Q LEARNING UPDATE FUNCTION
# -----------------------------

def update_q(state_index, action_index, reward):

    learning_rate = 0.1
    discount_factor = 0.9

    old_value = st.session_state.q_table[state_index, action_index]
    next_max = np.max(st.session_state.q_table[state_index])

    new_value = old_value + learning_rate * (
        reward + discount_factor * next_max - old_value
    )

    st.session_state.q_table[state_index, action_index] = new_value

    # Epsilon decay
    if st.session_state.epsilon > 0.05:
        st.session_state.epsilon *= 0.99


# -----------------------------
# UI
# -----------------------------

st.title("🤖 Smart Notification System (RL Based)")
st.write("The system learns what notification you like at different times.")

selected_time = st.selectbox("Select Time of Day", states)

state_index = states.index(selected_time)

# -----------------------------
# SEND NOTIFICATION BUTTON
# -----------------------------

if st.button("Send Smart Notification"):

    # Epsilon-Greedy
    if random.uniform(0, 1) < st.session_state.epsilon:
        action_index = random.randint(0, len(actions) - 1)
    else:
        action_index = np.argmax(st.session_state.q_table[state_index])

    chosen_action = actions[action_index]

    st.session_state.last_action = action_index
    st.session_state.last_state = state_index

    st.subheader("System Chose:")
    st.write(chosen_action)

    if chosen_action == "No Notification":
        st.info("No notification sent.")
    else:
        message = random.choice(notifications[selected_time][chosen_action])
        st.success(message)


# -----------------------------
# FEEDBACK SECTION
# -----------------------------

if st.session_state.last_action is not None:

    st.write("Did you engage with the notification?")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("👍 Yes"):
            update_q(
                st.session_state.last_state,
                st.session_state.last_action,
                reward=10
            )
            st.success("Positive reward given!")

    with col2:
        if st.button("👎 No"):
            update_q(
                st.session_state.last_state,
                st.session_state.last_action,
                reward=-5
            )
            st.error("Negative reward given!")


# -----------------------------
# DISPLAY Q TABLE
# -----------------------------

st.subheader("📊 Q Table")
st.dataframe(
    st.session_state.q_table,
    use_container_width=True
)

st.write("Current Epsilon:", round(st.session_state.epsilon, 3))
