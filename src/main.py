from utils.models import Models


test_questions = [
    "How do I dispute a charge I don't recognize on my statement?",
    "What information do I need to provide for a dispute?",
    "How long does it take to resolve a disputed transaction?",
    "Can I dispute a charge made by a subscription service?",
    "What should I do if I suspect fraud on my account?",
    "Will I get a refund for a transaction dispute?",
    "Can I dispute a cash withdrawal I didn't make?",
    "What happens if the merchant rejects my dispute?",
    "Are there any fees for disputing a transaction?",
    "How do I track the status of my dispute?"
]

models = Models()

for question in test_questions:
    base_response = models.customer_service_chat(
        question
    )
    ft_response = models.customer_service_chat(
        user = question, model="ft:gpt-4o-mini:bank-dispute-ft:8Y8888888888888888888888"
    )
