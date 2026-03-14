from preddict import predict_spam

while True:

    email = input("Enter email: ")
    result = predict_spam(email)
    print("Prediction:", result)
