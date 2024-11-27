import torch
import torch.nn as nn
import torch.optim as optim

# Ampliación de respuestas positivas y negativas
positive_responses = [
    "sí", "sip", "sipo", "ajá", "claro", "por supuesto", "seguro", 
    "sí, claro", "obvio", "yep", "yeah", "yup", "definitivamente", 
    "afirmativo", "correcto", "eso es"
]
negative_responses = [
    "no", "nope", "nop", "nel", "nunca", "jamás", "nah", 
    "imposible", "para nada", "ni de chiste", "negativo"
]

# Dataset extendido de respuestas y etiquetas (1 = positiva, 0 = negativa)
responses = positive_responses + negative_responses
labels = [1] * len(positive_responses) + [0] * len(negative_responses)

# Tokenización básica (convertir palabras a índices)
vocab = {word: i for i, word in enumerate(set(" ".join(responses).split()))}
tokenized_responses = [[vocab[word] for word in response.split()] for response in responses]

# Padding para que todas las secuencias tengan la misma longitud
max_len = max(len(seq) for seq in tokenized_responses)
padded_responses = [seq + [0] * (max_len - len(seq)) for seq in tokenized_responses]

# Convertir datos a tensores
X = torch.tensor(padded_responses, dtype=torch.long)
y = torch.tensor(labels, dtype=torch.float32)

# Modelo básico de red neuronal
class SimpleClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SimpleClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x).mean(dim=1)  # Promedio de los embeddings
        hidden = torch.relu(self.fc(embedded))
        out = self.sigmoid(self.output(hidden))
        return out

# Configuración del modelo
vocab_size = len(vocab)
embedding_dim = 10
hidden_dim = 5
model = SimpleClassifier(vocab_size, embedding_dim, hidden_dim)

# Configuración del optimizador y pérdida
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Entrenamiento
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X).squeeze()
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

# Función para predecir
def predict_responses(new_responses):
    tokenized = [[vocab.get(word, 0) for word in response.split()] for response in new_responses]
    padded = [seq + [0] * (max_len - len(seq)) for seq in tokenized]
    X_new = torch.tensor(padded, dtype=torch.long)
    predictions = model(X_new).squeeze()
    return [int(pred > 0.5) for pred in predictions]

# Prueba con nuevas respuestas
test_responses = ["sí", "no", "nunca", "quizás", "definitivamente", "sip", "nah", "obvio", "jamás"]
predictions = predict_responses(test_responses)
print("Predicciones:", predictions)
