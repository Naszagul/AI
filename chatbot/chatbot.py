import numpy as np
import datetime
import transformers as tf

class ChatBot():
    def __init__(self, name):
        print("----- Starting up", name, "-----")
        self.name = name
    def wake_up(self, text):
        return True if self.name in text.lower() else False
    def get_text(self):
        self.text = input("User: ")
    @staticmethod
    def action_time():
        return datetime.datetime.now().time().strftime('%H:%M')

# Running the AI
if __name__ == "__main__":
    ai = ChatBot(name="dev")
    model = tf.AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    tokenizer = tf.AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    nlp = tf.pipeline("conversational", model="microsoft/DialoGPT-medium")
    ex=True
    while ex:
        ai.get_text()
        ## action time
        if "time" in ai.text:
            res = f"Current time: {ai.action_time()} MST"
        elif ai.text in ["exit","close"]:
            salutation=["Goodbye!", "Auf Wiedersehen!", "Tschuss!", "Later!", "Bye!", "Take care!", "See you next time!"]
            res = np.random.choice(salutation)
            ex=False
        ## conversation
        else:   
            chat = nlp(tf.Conversation(ai.text), pad_token_id=50256)
            res = str(chat)
            res = res[res.find("bot >> ")+6:].strip()
        print(f"{ai.name}: {res}")
    print(f"----- Shutting down {ai.name} -----")