document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('spam-form');
    const result = document.getElementById('result');
    const loading = document.getElementById('loading');
    const historyList = document.getElementById('history-list');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const message = document.getElementById('message').value;
        const submitBtn = form.querySelector('button[type="submit"]');
        
        submitBtn.disabled = true;
        result.className = 'result';
        result.textContent = '';
        loading.style.display = 'flex';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/x-www-form-urlencoded'},
                body: `message=${encodeURIComponent(message)}`
            });
            const data = await response.json();
            const confidenceScore = (data.confidence * 100).toFixed(2);
            const resultText = `Result: ${data.result} (Confidence: ${confidenceScore}%)`;
            result.textContent = resultText;
            result.className = `result show ${data.result.toLowerCase().replace(' ', '-')}`;

            // Add to history
            const historyItem = document.createElement('li');
            historyItem.textContent = `${resultText} - ${new Date().toLocaleString()}`;
            historyList.insertBefore(historyItem, historyList.firstChild);

            // Limit history to 5 items
            if (historyList.children.length > 5) {
                historyList.removeChild(historyList.lastChild);
            }
        } catch (error) {
            console.error('Error:', error);
            result.textContent = 'An error occurred. Please try again.';
            result.className = 'result show';
        } finally {
            submitBtn.disabled = false;
            loading.style.display = 'none';
        }
    });
});