from flask import Flask, render_template, request

app = Flask(__name__)

# Render the HTML template
@app.route('/')
def index():
    return render_template('index.html')

# Handle the chatbot interaction
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form['message']
    # Process the user message and get a response from the chatbot logic
    # Replace the logic below with your actual chatbot's response
    bot_response = f"Your message: {user_message} (Bot response: I'm a simple chatbot!)"

    return bot_response

if __name__ == '__main__':
    app.run(debug=True)
