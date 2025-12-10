import google.generativeai as genai

API_KEY = "AIzaSyAyqeuJ-OQzubdsFAbNEbbrIk6Vz5g4I-0"
genai.configure(api_key=API_KEY)

model = genai.GenerativeModel("models/gemini-2.5-flash")

def main():
    print("Gemini Terminal Chatbot")
    print("Type 'exit' or 'quit' to end the conversation.\n")

    history = []

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit"]:
            print("Bot: Bye jaanu")
            break

        # Adding user message
        history.append({
            "role": "user",
            "parts": [user_input]
        })

        # Generating response with history
        response = model.generate_content(history)
        bot_reply = response.text.strip()

        # Adding model response to history
        history.append({
            "role": "model",
            "parts": [bot_reply]
        })

        print("Bot:", bot_reply)
        print()

if __name__ == "__main__":
    main()
