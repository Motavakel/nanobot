from nanochat.chatbot import NanoChat

if __name__ == "__main__":
    dataset = [
        ("hi", "Hello!"),
        ("hello", "Hi there!"),
        ("how are you", "I'm good, thanks!"),
        ("what is your name", "I'm NanoChat."),
        ("bye", "Goodbye!"),
    ]
    bot = NanoChat(dataset)
    bot.chat()
