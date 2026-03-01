import streamlit as st
import numpy as np
import random

# -------------------------
# STATES AND ACTIONS
# -------------------------

states = ["Morning", "Afternoon", "Evening"]
actions = ["News", "Social", "Market", "Entertainment", "No Notification"]

# -------------------------
# NOTIFICATIONS
# -------------------------

notifications = {
    "Morning": {
        "News": ["Good Morning! Here are today’s top headlines.", "Weather Update: Expect light showers today."],
        "Social": ["You have 5 new notifications.", "Your friend tagged you in a post."],
        "Market": ["Stock Market Opening Update.", "Nifty expected to rise today."],
        "Entertainment": ["Start your day with trending songs.", "Daily horoscope is ready."]
    },
    "Afternoon": {
        "News": ["Midday News Briefing.", "Live updates on current events."],
        "Social": ["Someone reacted to your story.", "You have unread messages."],
        "Market": ["Midday Market Update.", "Stock Alert: Price fluctuation detected."],
        "Entertainment": ["Trending videos you may like.", "Lunch break watch recommendations."]
    },
    "Evening": {
        "News": ["Evening Headlines Summary.", "Top stories you missed today."],
        "Social": ["Your post is getting attention.", "New comments on your photo."],
        "Market": ["Market Closing Report.", "Daily profit/loss summary."],
        "Entertainment": ["Your favorite show just dropped a new episode.", "Recommended movies for tonight."]
    }
}

# -------------------------
# SESSION STATE INIT
# -------------------------

if "q_table" not in st.session_state:
    st.session_state.q_table = np.zeros((len(states), len(actions)))

if "epsilon" not in st.session_state:
    st.session_state.epsilon = 1.0

learning_rate = 0.1
discount_factor = 0.9
epsilon_decay = 0.99
min_epsilon = 0.05

st.title("Smart RL Notification System")

# -------------------------
# TIME SELECTION
# -------------------------

selected_time = st.selectbox("Select Time of Day", states)

state_index = states.index(selected_time)

# -------------------------
# SEND NOTIFICATION
# -------------------------

if st.button("Send Smart Notification"):

    # Epsilon-Greedy
    if random.uniform(0, 1) < st.session_state.epsilon:
        action_index = random.randint(0, len(actions) - 1)
    else:
        action_index = np.argmax(st.session_state.q_table[state_index])

    chosen_action = actions[action_index]

    st.session_state.last_action_index = action_index
    st.session_state.last_state_index = state_index

    st.subheader("System Decision: " + chosen_action)

    if chosen_action == "No Notification":
        st.write("No notification sent.")
        st.session_state.reward = 0
    else:
        message = random.choice(notifications[selected_time][chosen_action])
        st.success(message)

# -------------------------
# FEEDBACK SECTION
# -------------------------

if "last_action_index" in st.session_state:

    col1, col2 = st.columns(2)

    if col1.button("Engaged 👍"):
        reward = 10
        update_q(reward=True)

    if col2.button("Ignored 👎"):
        reward = -5
        update_q(reward=False)


# -------------------------
# Q UPDATE FUNCTION
# -------------------------

def update_q(reward):

    state_index = st.session_state.last_state_index
    action_index = st.session_state.last_action_index

    if reward:
        r = 10
    else:
        r = -5

    old_value = st.session_state.q_table[state_index, action_index]
    next_max = np.max(st.session_state.q_table[state_index])

    new_value = old_value + learning_rate * (r + discount_factor * next_max - old_value)

    st.session_state.q_table[state_index, action_index] = new_value

    if st.session_state.epsilon > min_epsilon:
        st.session_state.epsilon *= epsilon_decay

    st.success("Model Updated Successfully!")

# -------------------------
# DISPLAY Q TABLE
# -------------------------

st.subheader("Current Q Table")
st.write(st.session_state.q_table)
