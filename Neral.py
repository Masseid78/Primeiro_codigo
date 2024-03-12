from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from iqoptionapi.stable_api import IQ_Option
import time 
import matplotlib.pyplot as plt
import numpy as np
# Conectando à IQ Option
I_want_money = IQ_Option("E-mail", "Senha")
I_want_money.connect()

# Verificando se a conexão foi bem-sucedida
if I_want_money.check_connect():
    print('Conexão bem-sucedida!')
else:
    print('Erro na conexão.')
    I_want_money.connect()

# Tentativa de carregar modelo anterior
try:
    model = load_model("meu_modelo.h5")
    print("Modelo anterior carregado com sucesso!")
except:
    model = None

# Loop de aprendizado contínuo (aqui, usando um for como exemplo para 30 dias)
for day in range(30):
    print(f"Treinando o dia {day+1}")
    
# Coletando dados dos pares de moedas 
timeframe = 60 * 60 * 24
end_from_time = time.time()
two_years_in_seconds = 60 * 60 * 24 * 365 * 8

pairs = ['EURUSD', 'USDJPY', 'GBPUSD', 'USDCHF',  'AUDUSD', 'USDCAD', 'NZDUSD', 'EURGBP', 'EURJPY']
all_data = []

for pair in pairs:
    data = I_want_money.get_candles(pair, timeframe, int((end_from_time - two_years_in_seconds) / timeframe), end_from_time)
    all_data.extend(data)

df = pd.DataFrame(all_data)

    
df = pd.DataFrame(data) 
print(df.columns)

# Convertendo a coluna de tempo para o formato datetime
df['time'] = pd.to_datetime(df['from'], unit='s')

# %K e %D (Estocástico)
low_min = df['min'].rolling(window=14).min()
high_max = df['max'].rolling(window=14).max()
df['%K'] = (df['close'] - low_min) / (high_max - low_min) * 100
df['%D'] = df['%K'].rolling(window=3).mean()

# Volatilidade
df['returns'] = df['close'].pct_change()
df['volatility'] = df['returns'].rolling(window=14).std()

# Momentum
df['momentum'] = df['close'] - df['close'].shift(4)
    
# RSI
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).fillna(0)
loss = (-delta.where(delta < 0, 0)).fillna(0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))

# Médias Móveis
df['MA5'] = df['close'].rolling(window=5).mean()
df['MA10'] = df['close'].rolling(window=10).mean()

# Bollinger Bands
df['MA20'] = df['close'].rolling(window=20).mean()
df['std20'] = df['close'].rolling(window=20).std()
df['UpperBB'] = df['MA20'] + (df['std20'] * 2)
df['LowerBB'] = df['MA20'] - (df['std20'] * 2)

# MACD
short_ema = df['close'].ewm(span=12, adjust=False).mean()
long_ema = df['close'].ewm(span=26, adjust=False).mean()
df['MACD'] = short_ema - long_ema
df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()


# EMA
df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()

#ON-BALANCE VOLUME
df['OBV'] = (df['volume'] * (~df['close'].diff().le(0) * 2 - 1)).cumsum()


# Removendo linhas com valores NaN
df = df.dropna()

# Coluna 'target'
df['target'] = df['close'].diff().apply(lambda x: 2 if x > 0 else (0 if x < 0 else 1))


# Especificando as colunas dos indicadores no DataFrame para serem usadas como features
print(df.columns)
X = df[['RSI', 'MA5', 'MA10', 'MA20', 'EMA20', 'UpperBB', 'LowerBB', 'MACD', 'Signal_Line', '%K', '%D', 'volatility', 'momentum', 'OBV']]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizando os dados
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Remodelando os dados para a forma que a CNN 1D espera: [samples, timesteps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

if model is None: 
    model = Sequential([

    Dense(1500, activation='relu', input_dim=X_train.shape[1]),  # Primeira camada de neuronios
    Dropout(0.4),  # Dropout de 40%
    
    Dense(500, activation='relu'),   # segunda camada de neuronios
    Dropout(0.4),  # Dropout de 40%
    
    Dense(500, activation='relu'),    # terceira camada de neuronios
    Dropout(0.4),  # Dropout de 40%
    
    Dense(125, activation='relu'),    # quarta camada de neuronios
    Dropout(0.4),  # Dropout de 40%
    
    Dense(100, activation='relu'),    # quinta camada de neuronios
    Dropout(0.4),  # Dropout de 40%
  
    Dense(50, activation='softmax')  
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

model.save("meu_modelo.h5")
print(f"Modelo atualizado e salvo no dia {day+1}")

time.sleep(60*60*24)