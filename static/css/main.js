const chatArea = document.getElementById('chatArea');
const userInput = document.getElementById('userInput');
const loading = document.getElementById('loading');

// Send on Enter key
userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') sendMessage();
});

function addMessage(text, sender) {
    const msg = document.createElement('div');
    msg.className = `message ${sender}`;
    msg.innerHTML = `<div class="bubble">${text}</div>`;
    chatArea.insertBefore(msg, loading);
    chatArea.scrollTop = chatArea.scrollHeight;
}

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    addMessage(text, 'user');
    userInput.value = '';
    loading.classList.add('active');

    const memoryLog = document.getElementById('memoryLog');
    userInput.value = '';
    document.getElementById('memoryPanel').style.display = 'block';

    const evtSource = new EventSource('/memory/stream')
    evtSource.onmessage = (event) => {
        const step = JSON.parse(event.data);
        memoryLog.innerHTML = `
            <div class="memory-entry">
                <strong>Step ${step.step}: </strong>
                <div class="memory-thought">${step.memory || 'No memory'}</div>
            </div>
        `;
        // const entry = document.createElement('div');
        // entry.className = 'memory-entry';
        // entry.innerHTML = `
        //     <strong>Step ${step.step}:</strong>
        //     <div class="memory-thought">${step.memory || 'No memory'}</div>
            
        // `;
        // memoryLog.appendChild(entry);
        // memoryLog.scrollTop = memoryLog.scrollHeight;
    }

    try {
        const res = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text })
        });
        const data = await res.json();
        loading.classList.remove('active');
        addMessage(data.response || 'No response.', 'bot');
    } catch (err) {
        loading.classList.remove('active');
        addMessage('Server error. Is the backend running?', 'bot');
    }
}


