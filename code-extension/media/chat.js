(function () {
    const vscode = acquireVsCodeApi();
    const input = document.getElementById('input');
    const send = document.getElementById('send');
    const messages = document.getElementById('messages');

    // Helper to append a message
    function appendMessage(text, from) {
        const div = document.createElement('div');
        div.className = 'message ' + from;
        div.textContent = text;
        messages.appendChild(div);
        messages.scrollTop = messages.scrollHeight;
    }

    send.addEventListener('click', () => {
        const text = input.value.trim();
        if (!text) return;
        appendMessage(text, 'user');
        vscode.postMessage({ command: 'sendMessage', text });
        input.value = '';
    });

    // Listen for messages FROM the extension
    window.addEventListener('message', event => {
        const msg = event.data;
        if (msg.command === 'newMessage') {
            appendMessage(msg.text, msg.from);
        }
    });
})();
