from dotenv import load_dotenv
from sqlalchemy import create_engine
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import f_classif,SelectKBest
from sklearn.model_selection import train_test_split


# load the .env file variables
load_dotenv()


def db_connect():
    import os
    engine = create_engine(os.getenv('DATABASE_URL'))
    engine.connect()
    return engine

def columnas_categoricas(df, lista_num_cat=[]):
    columnas_categoricas = []
    for column in df.columns:
        if df[column].dtype == 'object':
            columnas_categoricas.append(column)
    if len(lista_num_cat)>0:
        columnas_categoricas.extend(lista_num_cat)
    return columnas_categoricas

def analisis_categorico_categorico(df,lista_num_cat=[]):
    col_cat = columnas_categoricas(df,lista_num_cat)
    largo = len(col_cat)
    parte_entera = largo//3
    resto = largo%3
    parte_entera+=1
    if resto == 0:
        parte_entera -=1
    
    fig, axis = plt.subplots(parte_entera,3,figsize=(10,7))

    for column in col_cat:
        fila=col_cat.index(column)//3
        columna = col_cat.index(column)%3
        if columna != 0:
            sns.histplot(ax= axis[fila,columna],data=df,x=column).set(ylabel=None,xticks=[])
        else:
            sns.histplot(ax= axis[fila,columna],data=df,x=column).set(xticks=[])

    plt.tight_layout()
    plt.show()

def columnas_numericas(df,columnas_excluidas=[]):
    columnas_numericas = []
    for column in df.columns:
        if df[column].dtype in ['int64','float64']:
            if column not in columnas_excluidas:
                columnas_numericas.append(column)  
    return columnas_numericas

def analisis_numerico_numerico(df,y,columnas_excluidas=[],limites=[]):
    col_num = columnas_numericas(df,columnas_excluidas)
    if df[y].dtype not in ['int64','float64']:
        col_num.remove(y)
    largo = len(col_num)
    parte_entera = (largo//2)*2
    resto = largo%4
    parte_entera +=2
    if resto == 0:
        parte_entera-=2
    num_de_gridspec = math.ceil(parte_entera/2)
    lista_hr = [5,1]*num_de_gridspec
    fig, axis = plt.subplots(parte_entera,2,figsize=(10,10),gridspec_kw={"height_ratios":lista_hr})

    for column in col_num:
        fila = (col_num.index(column)//2)*2
        columna = col_num.index(column)%2
        if len(limites)>0:
            if 0<= col_num.index(column)*2 < len(limites):
                if limites[col_num.index(column)*2] != None:
                    sns.histplot(ax=axis[fila,columna],data=df,x=column).set(xlim=(limites[col_num.index(column)*2],limites[col_num.index(column)*2+1]))
                    sns.boxplot(ax=axis[fila+1,columna],data=df,x=column).set(xlim=(limites[col_num.index(column)*2],limites[col_num.index(column)*2+1]))
                else:
                    sns.histplot(ax=axis[fila,columna],data=df,x=column)
                    sns.boxplot(ax=axis[fila+1,columna],data=df,x=column)
            else:
                sns.histplot(ax=axis[fila,columna],data=df,x=column)
                sns.boxplot(ax=axis[fila+1,columna],data=df,x=column)
        else:
            sns.histplot(ax=axis[fila,columna],data=df,x=column)
            sns.boxplot(ax=axis[fila+1,columna],data=df,x=column)
    plt.tight_layout()
    plt.show()

def analisis_categorico_categorico_multivariante(df,columna1,columna2):
    sns.countplot(data=df,x=columna1,hue=columna2)
    plt.show()

def analisis_numerico_numerico_multivariante(df,y,columnas_excluidas=[]):
    col_num = columnas_numericas(df,columnas_excluidas)
    col_num.remove(y)
    largo = len(col_num)
    parte_entera = (largo//2)*2
    resto = largo%4
    parte_entera +=2
    if resto == 0:
        parte_entera-=2
    fig, axis = plt.subplots(parte_entera,2,figsize=(10,16))

    for column in col_num:
        fila = (col_num.index(column)//2)*2
        columna = col_num.index(column)%2
        sns.regplot(ax=axis[fila,columna], data=df, x=column, y=y)
        sns.heatmap(df[[y,column]].corr(),annot=True,fmt='.2f',ax=axis[fila+1,columna],cbar=False)
    plt.tight_layout()
    plt.show()

def analisis_numerico_categorico(df,columnas_excluidas=[]):
    if len(columnas_excluidas)>0:
        df2 = df.drop(columnas_excluidas,axis=1)
    else:
        df2 = df
    col_cat = columnas_categoricas(df2)
    for column in col_cat:
        df2[column] = pd.factorize(df2[column])[0]
    sns.heatmap(df2.corr(),annot=True,fmt='.2f')
    plt.show()

def analisis_boxplot(df,y,lista_num_cat=[]):
    df2 = df.drop([y],axis=1)
    col_num = columnas_numericas(df,lista_num_cat)
    col_cat = columnas_categoricas(df)

    for column in col_cat:
        df2[column] = pd.factorize(df2[column])[0]
    
    col_cat.extend(col_num)
    largo = len(col_cat)
    parte_entera = largo//3
    resto = largo%3
    parte_entera+=1
    if resto == 0:
        parte_entera -=1
    
    fig, axis = plt.subplots(parte_entera,3,figsize=(10,7))

    for column in col_cat:
        fila=col_cat.index(column)//3
        columna = col_cat.index(column)%3
        sns.boxplot(ax= axis[fila,columna],data=df,y=column).set(xticks=[])

    plt.tight_layout()
    plt.show()

def analisis_ric(df):
    col_num = columnas_numericas(df)
    for columna in col_num:
        estadisticas = df[columna].describe()
        ric = estadisticas['75%'] - estadisticas['25%']
        lim_sup = estadisticas['75%'] + 1.5*ric
        lim_inf = estadisticas['25%'] - 1.5*ric 
        print(f'columna: {columna} limites superior {lim_sup} e inferior {lim_inf}')

def escalar_caracteristicas(df,y,tipo_escalado='MinMax',columnas_excluidas=[]):
    if len(columnas_excluidas)>0:
        df2 = df.drop(columnas_excluidas,axis=1)
    else:
        df2 = df
    objetivo = df2[y]
    df2.drop([y],axis=1,inplace=True)    
    variables = columnas_categoricas(df2)
    col_num = columnas_numericas(df2)

    for columna in variables:
        df2[columna] = pd.factorize(df2[columna])[0]

    variables.extend(col_num)
    
    if tipo_escalado=='MinMax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    caracteristicas_escaladas = scaler.fit_transform(df2[variables])
    df_escalado = pd.DataFrame(caracteristicas_escaladas,index=df2.index, columns = variables)
    return df_escalado, objetivo

def seleccionar_mejores_caracteristicas(X,y,k=5,test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size, random_state = 42)

    modelo_seleccion = SelectKBest(f_classif,k=k)
    modelo_seleccion.fit(X_train,y_train)
    ix = modelo_seleccion.get_support()

    X_train_sel = pd.DataFrame(modelo_seleccion.transform(X_train),columns=X_train.columns.values[ix])
    X_test_sel = pd.DataFrame(modelo_seleccion.transform(X_test),columns=X_test.columns.values[ix])

    return X_train_sel, X_test_sel, y_train, y_test

def guardar_csv_limpios(X_train_sel, X_test_sel, y_train, y_test,carpeta='../data/processed/'):
    X_train_sel['price'] = list(y_train)
    X_test_sel['price'] = list(y_test)
    X_train_sel.to_csv(carpeta+'train_limpio.csv',index=False)
    X_test_sel.to_csv(carpeta+'test_limpio.csv',index=False)