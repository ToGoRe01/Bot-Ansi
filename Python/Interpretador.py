import torch
import torch.nn as nn
import torch.optim as optim
import json
import sys

node_data = sys.argv[1]
data_parsed = json.loads(node_data)
user_responses = data_parsed["results"]

# Respuestas posibles y pesos asignados
positive_responses = ["si", "sí", "sip", "sipo", "ajá", "claro", "por supuesto", "seguro", "sí, claro", "obvio", "definitivamente", "afirmativo"]
negative_responses = ["no", "nope", "nop", "nel", "nunca", "jamás", "nah", "imposible", "para nada", "ni de chiste", "negativo"]

# Etiquetas y dataset simulado
responses = positive_responses + negative_responses
labels = [1] * len(positive_responses) + [0] * len(negative_responses)

# Generar vocabulario a partir de todas las palabras
all_words = set(word for response in responses for word in response.split())
vocab = {word: i + 1 for i, word in enumerate(all_words)}  # Índices empiezan desde 1
vocab["<UNK>"] = 0  # Palabra desconocida

# Tokenización
tokenized_responses = [[vocab.get(word, 0) for word in response.split()] for response in responses]

# Padding
max_len = max(len(seq) for seq in tokenized_responses)
padded_responses = [seq + [0] * (max_len - len(seq)) for seq in tokenized_responses]

# Convertir datos a tensores
X = torch.tensor(padded_responses, dtype=torch.long)
y = torch.tensor(labels, dtype=torch.float32)

# Modelo básico
class SimpleClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SimpleClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x).mean(dim=1)
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
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}", file=sys.stderr)

# Función para predecir
def predict_responses(new_responses):
    tokenized = [[vocab.get(word, 0) for word in response.split()] for response in new_responses]
    padded = [seq + [0] * (max_len - len(seq)) for seq in tokenized]
    X_new = torch.tensor(padded, dtype=torch.long)
    predictions = model(X_new).squeeze()
    return [int(pred > 0.5) for pred in predictions]

# Evaluación con porcentaje asignado a cada pregunta
def evaluate_user_responses(user_responses):
    percentages = [8, 10, 7, 5, 10, 5, 8, 7, 10, 12, 8, 5, 5]  # Porcentajes asignados a cada pregunta
    predictions = predict_responses(user_responses)
    positive_percentage = sum(percentages[i] for i in range(len(user_responses)) if predictions[i] == 1)
    total_percentage = sum(percentages)
    
    # Calcular nivel de ansiedad
    stress_level = (positive_percentage / total_percentage) * 100
    if stress_level < 30:
        anxiety_status = "Bajo"
    elif 30 <= stress_level < 60:
        anxiety_status = "Moderado"
    else:
        anxiety_status = "Alto"

    return stress_level, anxiety_status

# Evaluación con consejos amigables según el nivel de ansiedad
def evaluate_user_responses_with_advice(user_responses):
    percentages = [8, 10, 7, 5, 10, 5, 8, 7, 10, 12, 8, 5, 5]  # Porcentajes asignados a cada pregunta
    predictions = predict_responses(user_responses)
    positive_percentage = sum(percentages[i] for i in range(len(user_responses)) if predictions[i] == 1)
    total_percentage = sum(percentages)
    
    # Calcular nivel de ansiedad
    stress_level = (positive_percentage / total_percentage) * 100
    if stress_level < 30:
        anxiety_status = "Bajo"
        advice = (
            "¡Genial! Parece que te sientes tranquilo. Manten este estado disfrutando actividades que te gusten, "
            "como leer, caminar o escuchar música."
        )
    elif 30 <= stress_level < 60:
        anxiety_status = "Moderado"
        advice = (
            "Es posible que tengas algo de tensión. Considera tomar pequeños descansos, "
            "hacer ejercicios de respiración o practicar algún hobby que te relaje."
        )
    else:
        anxiety_status = "Alto"
        advice = (
            "Parece que podrías estar sintiendo bastante estrés. Te sugerimos cuidar de ti mismo: "
            "descansa lo suficiente, come bien y, si es necesario, habla con alguien de confianza sobre cómo te sientes."
        )
    
    return stress_level, anxiety_status, advice

# Simulación de respuestas del usuario
stress_level, anxiety_status, advice = evaluate_user_responses_with_advice(user_responses)

# Resultado
diagnostico = {
    "stress_level": stress_level,
    "anxiety_status": anxiety_status,
    "advice": advice
}
print(json.dumps(diagnostico))

