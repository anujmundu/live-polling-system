# Live Polling System

A secure, modular real-time polling app for teachers and students. Built with **React**, **Express**, and **Socket.IO**, it supports encrypted authentication, role-based flows, and robust poll lifecycle enforcement.

---

## Features

- 🔐 Encrypted token-based socket authentication
- 🧑‍🏫 Role-based flows for teacher and student
- ⏱️ Poll lifecycle enforcement (all answered or 60s timeout)
- 📦 Modular backend logic (`pollManager.js`, `socket/index.js`)
- ⚙️ CRA frontend with environment variable support
- 📡 Real-time updates via Socket.IO

---

## 🛠️ Local Setup

### 1. Clone the Repo

```bash
git clone https://github.com/anujmundu/live-polling-system.git
cd live-polling-system

Backend Setup

cd server
npm install
npm start

Make sure to set environment variables in server/.env:

PORT=5000
SOCKET_SECRET=your-secret

Frontend Setup

cd ../client
npm install
npm start

Set frontend environment variables in client/.env:

REACT_APP_SOCKET_SECRET=your-secret
REACT_APP_SOCKET_URL=http://localhost:5000

Usage
Teacher creates a poll and shares the room code.

Students join using the code and submit answers.

Poll ends when all students answer or after 60 seconds.

Results are broadcast in real-time.
