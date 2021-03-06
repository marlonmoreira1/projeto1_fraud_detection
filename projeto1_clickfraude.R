#carregando os pacotes necessarios
library(data.table)
library(C50)
library(caret)
library(ROCR)
library(pROC)
library(ROSE)
library(lubridate)
library(caTools)
library(rpart)
library(e1071)

#coletando os dados
treino = fread('treino.csv',stringsAsFactors = F, sep = ',',header = T)
teste = fread('teste.csv',nrow=100000,stringsAsFactors = F, sep = ',',header = T)

#visualizando os dados
str(treino)
View(treino)
dim(treino)

#verificando se há valor ausente
sum(is.na(treino))


#visualizando os dados
str(teste)
View(teste)
dim(teste)

#verificando se há valor ausente
sum(is.na(teste))

#Não há registros nessa coluna (optei por apaga-la)
treino$attributed_time = NULL

#Todas as outras variaveis são do tipo inteiro, então decidir 
#transforma a data do click em inteira tambem
treino$click_time = ymd_hms(treino$click_time)
treino$click_time = decimal_date(treino$click_time)
treino$click_time = as.integer(treino$click_time)

teste$click_time = ymd_hms(teste$click_time)
teste$click_time = decimal_date(teste$click_time)
teste$click_time = as.integer(teste$click_time)

#olhando aproporção da variavel target
prop.table(table(treino$is_attributed))

treino$is_attributed = as.factor(treino$is_attributed)

#fazendo o balanceamento
novo_treino = ROSE(is_attributed~.,data=treino,seed = 1)$data

prop.table(table(novo_treino$is_attributed))

sum(is.na(novo_treino))

#criando o modelo
modelo = C5.0(is_attributed~.,data = novo_treino)

#fazeendo a previsão 
previsao = predict(modelo,teste)

#verificando a acuracia
caret::confusionMatrix(novo_treino$is_attributed,previsao,positive = '1')

#Calculando o Score AUC
roc.curve(novo_treino$is_attributed,previsao,plotit = T,col='red')

#Eu não entendi porque o dataset de teste não há a variavel target
#Por isso, decidi dividir o dataset de treino e criar um novo modelo
amostra = sample.split(novo_treino,SplitRatio = 0.70)

train = subset(novo_treino,amostra==TRUE)
test = subset(novo_treino,amostra==FALSE)

modelo2 = C5.0(is_attributed~.,data = train)

previsao2 = predict(modelo2,test)

caret::confusionMatrix(test$is_attributed,previsao2,positive='1')

roc.curve(test$is_attributed,previsao2,plotit = T,col='green')


#testando com outro algoritmo
modelo3 = svm(is_attributed~.,
              data = train,
              type = 'C-classification',
              kernel = 'radial')

previsao3 = predict(modelo3,test)

mean(previsao3==test$is_attributed)

table(previsao3,test$is_attributed)

prop.table(table(train$is_attributed))
prop.table(table(test$is_attributed))


#testando sem a coluna de data do click
train$click_time = NULL
test$click_time = NULL

modelo4 = svm(is_attributed~.,
              data = train,
              type='C-classification',
              kernel='radial')

previsao4 = predict(modelo4,test)

caret::confusionMatrix(test$is_attributed,previsao4,positive='1')

roc.curve(test$is_attributed,previsao4,plotit=T,col='blue')

