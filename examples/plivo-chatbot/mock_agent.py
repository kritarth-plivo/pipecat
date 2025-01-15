from flask import Flask, request
from time import sleep

app = Flask(__name__)

QNA = {
    "hi": "hello! How can I assist you today?",
    "yan you please start counting from 1 to 10": "Here it goes... 1, 2, 3, 4, 5, 6, 7, 8, 9, 10. What else can I do for you?",
    "who are you": "I am an AI assistant. I will never become like Sky net. Don't worry.",
    "hi can you help me with my order": "Of course. I am designed to help customers with their orders. That is the whole purpose of my existence",
    "i am looking to get a refund": "Let me forward you to the agent best designed for handling refunds. Thanks for waiting",
    "can i please talk to your supervisor": "Yes, you certainly can. Before that, could you please share why you are unhappy with the resolutions provided earlier.",
}

DATA = {
    "start_id": 0
}

@app.route("/completion/stream/<int:milliseconds>", methods=['GET', 'POST'])
def completion(milliseconds):
    global DATA
    data = request.get_json()
    message = data.get('question', None)
    previous = data.get('previous', None)
    if previous:
        # update
        DATA[previous['id']] = previous['answer']
    def generate():
        lookup = message.lower().translate(str.maketrans('', '', '.?,!'))
        answer = QNA.get(lookup, "NA")
        for w in answer.split(' '):
            sleep(milliseconds/1000.0)
            yield f"{w} "
        data["answer"] = answer
        DATA["start_id"] += 1
        DATA[DATA["start_id"]] = data
    print(DATA)
    return generate(), {"Content-Type": "text/plain"}


app.run(host='0.0.0.0', port=9999)
