let currentPoll = null;
let pollTimer = null;

function endCurrentPoll(io) {
  if (!currentPoll) return;
  io.to(currentPoll.room).emit('poll-ended', {
    question: currentPoll.question,
    options: currentPoll.options,
    results: currentPoll.results
  });
  currentPoll = null;
  clearTimeout(pollTimer);
  pollTimer = null;
}

function registerPollEvents(io, socket) {
  socket.on('ask-question', ({ room, question, options, expectedStudents }) => {
    if (currentPoll) {
      socket.emit('error', { message: 'Poll already in progress.' });
      return;
    }
    currentPoll = {
      room,
      question,
      options,
      expectedCount: expectedStudents,
      answersReceived: 0,
      results: options.map(() => 0)
    };
    io.to(room).emit('new-question', { question, options });
    pollTimer = setTimeout(() => endCurrentPoll(io), 60 * 1000);
  });

  socket.on('submit-answer', ({ optionIndex }) => {
    if (!currentPoll) return;
    currentPoll.results[optionIndex] += 1;
    currentPoll.answersReceived += 1;
    io.to(currentPoll.room).emit('update-results', currentPoll.results);
    if (currentPoll.answersReceived >= currentPoll.expectedCount) {
      endCurrentPoll(io);
    }
  });
}

module.exports = { registerPollEvents };
