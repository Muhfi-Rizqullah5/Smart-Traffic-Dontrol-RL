import os
import sys
import traci
import sumolib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from collections import deque
import random

# --- KONFIGURASI SUMO ---
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

SUMO_CMD = ["sumo", "-c", "configuration/intersection.sumocfg"] # Pastikan path config benar

# --- PARAMETER DQN ---
STATE_SIZE = 4 # Contoh: Jumlah jalur masuk
ACTION_SIZE = 4 # Contoh: Jumlah fase lampu
MEMORY_SIZE = 2000
GAMMA = 0.95    # Discount rate
EPSILON = 1.0  # Exploration rate
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001
BATCH_SIZE = 32

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON
        self.model = self._build_model()

    def _build_model(self):
        # Neural Network Sederhana untuk Deep Q-Learning
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size) # Explore
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0]) # Exploit

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + GAMMA * np.amax(self.model.predict(next_state, verbose=0)[0]))
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

def get_state():
    # Fungsi placeholder untuk mengambil data dari sensor SUMO (Induction Loops)
    # Contoh: Mengambil jumlah kendaraan yang berhenti di setiap jalur
    queue_length_lane_1 = traci.lane.getLastStepHaltingNumber("lane1_id")
    queue_length_lane_2 = traci.lane.getLastStepHaltingNumber("lane2_id")
    # ... dst
    return np.array([[queue_length_lane_1, queue_length_lane_2, 0, 0]]) # Sesuaikan dimensi

def run_simulation():
    traci.start(SUMO_CMD)
    agent = DQNAgent(STATE_SIZE, ACTION_SIZE)
    step = 0
    
    while step < 1000: # Loop simulasi
        traci.simulationStep()
        
        # 1. Dapatkan State saat ini
        current_state = get_state()
        
        # 2. Agen memilih aksi (Fase lampu hijau)
        action = agent.act(current_state)
        
        # 3. Terapkan aksi ke Traffic Light SUMO
        traci.trafficlight.setPhase("TL_ID", action)
        
        # 4. Hitung Reward (Misal: -Total Waiting Time)
        waiting_time = traci.edge.getWaitingTime("edge_id")
        reward = -waiting_time 
        
        # 5. Dapatkan Next State & Train
        next_state = get_state()
        agent.remember(current_state, action, reward, next_state, False)
        agent.replay(BATCH_SIZE)
        
        step += 1
        
    traci.close()

if __name__ == "__main__":
    run_simulation()
