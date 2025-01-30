function sendMessage() {
    const userInput = document.getElementById("user-input").value;
    if (userInput.trim() === "") return; // Prevent empty messages

    // Add the user's message to the chatbox
    const chatBox = document.getElementById("chat-box");
    const userMessageDiv = document.createElement("div");
    userMessageDiv.classList.add("user-message");
    userMessageDiv.innerHTML = `<p>${userInput}</p>`;
    chatBox.appendChild(userMessageDiv);

    // Scroll to the bottom of the chat
    chatBox.scrollTop = chatBox.scrollHeight;

    // Clear the input field
    document.getElementById("user-input").value = "";

    // Send the message to the backend
    $.ajax({
        url: '/api/chat',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ query: userInput }),
        success: function(response) {
            const botMessageDiv = document.createElement("div");
            botMessageDiv.classList.add("bot-message");
            botMessageDiv.innerHTML = `<p>${response.response}</p>`;
            chatBox.appendChild(botMessageDiv);

            // Scroll to the bottom of the chat
            chatBox.scrollTop = chatBox.scrollHeight;
        },
        error: function() {
            const botMessageDiv = document.createElement("div");
            botMessageDiv.classList.add("bot-message");
            botMessageDiv.innerHTML = `<p>Sorry, something went wrong. Please try again.</p>`;
            chatBox.appendChild(botMessageDiv);

            // Scroll to the bottom of the chat
            chatBox.scrollTop = chatBox.scrollHeight;
        }
    });
}
