from chatbot import ask_question

print("\nVayuVidh 🌱 CBG Assistant")
print("Type 'exit' to quit.\n")

while True:

    question = input("Ask a question: ")

    if question.lower() == "exit":
        break

    try:
        answer, sources = ask_question(question)

        print("\nANSWER\n")
        print(answer)

        print("\nSOURCES\n")

        for s in sources:
            print(f"- {s['source_file']} | Page {s['page']}")

        print("\n" + "="*80 + "\n")

    except Exception as e:
        print("\nERROR:")
        print(e)