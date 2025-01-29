async function sendMessage() {
    const input = document.getElementById('user-input');
    const history = document.getElementById('chat-history');
    const message = input.value.trim();

    if (!message) return;

    // Add user message
    history.innerHTML += `
        <div class="message user-message">
            <strong>You:</strong> ${message}
        </div>
    `;

    // Clear input
    input.value = '';

    // Get bot response
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({query: message})
        });

        const data = await response.json();

        // Add bot message
        history.innerHTML += `
            <div class="message bot-message">
                <strong>Bot:</strong> ${data.response}
            </div>
        `;

        // Scroll to bottom
        history.scrollTop = history.scrollHeight;
    } catch (error) {
        console.error('Error:', error);
    }
}

// Handle Enter key
document.getElementById('user-input').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});