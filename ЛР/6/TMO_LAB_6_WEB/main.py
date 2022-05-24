import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR


def about():
    st.header('Лабораторная работа №6')
    st.write('Чиварзин А. Е. ИУ5Ц-82Б')
    st.markdown('---------------------------------------------------------------------')


@st.cache
def load_data():
    '''
    Загрузка данных
    '''
    data_original = pd.read_csv('data/cwurData.csv', sep=",")
    return data_original


@st.cache
def delete_NULLs(data_in):
    data_out = data_in.copy()
    # Удаление строк, содержащих пустые значения
    data_out = data_out.dropna(axis=0, how='any')
    return data_out


@st.cache
def preprocess_data(data_in):
    '''
    Масштабирование и кодирование признаков, функция возвращает X и y для кросс-валидации
    '''
    data_out = data_in.copy()
    le = LabelEncoder()
    institution_le = le.fit_transform(data_out['institution'])
    le_country = LabelEncoder()
    country_le = le_country.fit_transform(data_out['country'])
    data_digit = data_out.copy()
    data_digit["institution"] = institution_le
    data_digit['country'] = country_le

    sc1 = MinMaxScaler()
    sc1_data = sc1.fit_transform(data_digit[['broad_impact']])
    sc2 = MinMaxScaler()
    sc2_data = sc2.fit_transform(data_digit[['institution']])
    sc3 = MinMaxScaler()
    sc3_data = sc3.fit_transform(data_digit[['country']])
    sc4 = MinMaxScaler()
    sc4_data = sc4.fit_transform(data_digit[['national_rank']])
    sc5 = MinMaxScaler()
    sc5_data = sc5.fit_transform(data_digit[['quality_of_education']])
    sc6 = MinMaxScaler()
    sc6_data = sc6.fit_transform(data_digit[['alumni_employment']])
    sc7 = MinMaxScaler()
    sc7_data = sc7.fit_transform(data_digit[['quality_of_faculty']])
    sc8 = MinMaxScaler()
    sc8_data = sc8.fit_transform(data_digit[['publications']])
    sc9 = MinMaxScaler()
    sc9_data = sc9.fit_transform(data_digit[['influence']])
    sc10 = MinMaxScaler()
    sc10_data = sc10.fit_transform(data_digit[['citations']])
    sc11 = MinMaxScaler()
    sc11_data = sc11.fit_transform(data_digit[['broad_impact']])
    sc12 = MinMaxScaler()
    sc12_data = sc12.fit_transform(data_digit[['patents']])
    sc13 = MinMaxScaler()
    sc13_data = sc13.fit_transform(data_digit[['score']])
    sc14 = MinMaxScaler()
    sc14_data = sc14.fit_transform(data_digit[['year']])

    data_normal = data_digit.copy()
    data_normal['world_rank'] = sc1_data
    data_normal['institution'] = sc2_data
    data_normal['country'] = sc3_data
    data_normal['national_rank'] = sc4_data
    data_normal['quality_of_education'] = sc5_data
    data_normal['alumni_employment'] = sc6_data
    data_normal['quality_of_faculty'] = sc7_data
    data_normal['publications'] = sc8_data
    data_normal['influence'] = sc9_data
    data_normal['citations'] = sc10_data
    data_normal['broad_impact'] = sc11_data
    data_normal['patents'] = sc12_data
    data_normal['score'] = sc13_data
    data_normal['year'] = sc14_data
    data_out = data_normal
    return pd.DataFrame(data_out).drop(['world_rank'], axis=1), data_out['world_rank']


#######################################################################################################################

def KNN(x, y):
    reg_gs = GridSearchCV(KNeighborsRegressor(), tuned_parameters, cv=cv_slider, scoring='neg_median_absolute_error')
    reg_gs.fit(x, y)

    st.subheader('Оценка качества модели')

    st.write('Лучшее значение параметров - {}'.format(reg_gs.best_params_))

    # Изменение качества на тестовой выборке в зависимости от К-соседей
    fig1 = plt.figure(figsize=(7, 5))
    ax = plt.plot(n_range, reg_gs.cv_results_['mean_test_score'])
    plt.xlabel('Количество соседей')
    st.pyplot(fig1)


def SVM(X: pd.DataFrame, Y):
    if X.shape[1] == 0:
        st.write('Ни один столбец не выбран')
    else:
        gp = st.selectbox('Гиперпараметр для построения', X.columns)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=2022, test_size=0.1)
        svr = SVR(kernel='linear')
        svr.fit(X_train, Y_train)
        pred_y = svr.predict(X_test)
        fig2 = plt.figure(figsize=(7, 5))
        plt.scatter(X_test[gp], Y_test, marker='s', label='Тестовая выборка')
        plt.scatter(X_test[gp], pred_y, marker='o', label='Предсказанные данные')
        plt.legend(loc='lower right')
        plt.xlabel('рейтинг за широкое влияние')
        plt.ylabel('Целевой признак')
        st.pyplot(fig2)


about()
use_msg = True
if os.name == 'nt':
    if st.sidebar.checkbox('Оповещать об окончании обучения', value=True):
        use_msg = True
    else:
        use_msg = False

data = load_data()
data_no_null = delete_NULLs(data)

if st.checkbox('Показать данные'):
    st.write(data_no_null)

if st.checkbox('Показать парные диаграммы (+ 30 секунд)'):
    st.pyplot(sns.pairplot(data_no_null, height=5))
    st.write('Разглядеть не получится. Известный баг: https://github.com/streamlit/streamlit/issues/796')

if st.checkbox('Показать корреляционную матрицу'):
    fig1, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(data_no_null.corr(), annot=True, fmt='.2f')
    st.pyplot(fig1)

st.markdown('--------------------------------------------------------------------------------------------------')

st.sidebar.header('Метод ближайших соседей')  #########################################################################
do_analize_KNN = False  # По-умолчанию отключим анализ данных, чтобы не ждать 15 секунд - 5 ммнут!!!
if st.sidebar.checkbox('Показать гиперпараметры'):
    cv_slider = st.sidebar.slider('Количество фолдов:', min_value=2, max_value=10, value=3, step=1)
    step_slider = st.sidebar.slider('Шаг для соседей:', min_value=1, max_value=50, value=10, step=1)
    # Количество записей
    data_len = data.shape[0]
    data_no_null_len = data_no_null.shape[0]
    # Вычислим количество возможных ближайших соседей
    rows_in_one_fold = int(data_no_null_len / cv_slider)
    allowed_knn = int(rows_in_one_fold * (cv_slider - 1))
    st.write('Количество строк в наборе данных до очистки строк - {}'.format(data_len))
    st.write('Количество строк в наборе данных после очистки строк - {}'.format(data_no_null_len))
    st.write('Максимальное допустимое количество ближайших соседей с учётом выбранного количества фолдов - {}'.format(allowed_knn))
    # Подбор гиперпараметра
    n_range_list = list(range(1, allowed_knn, step_slider))
    n_range = np.array(n_range_list)
    st.write('Возможные значения соседей - {}'.format(n_range))
    tuned_parameters = [{'n_neighbors': n_range}]

    if st.sidebar.checkbox('Выполнять обучение'):
        do_analize_KNN = True
    else:
        do_analize_KNN = False
    st.sidebar.write(
        '⚠ Устанавливайте этот флажок только после окончательной установки гиперпараметров\n'
        'После установки флажка рекоммендуется идти пить чай на 30 секунд - 30 минут...')

st.sidebar.header('SVM')  #############################################################################################
do_analize_SVM = False
show_SVM_params = False
if st.sidebar.checkbox('Показать гиперпараметры '):
    show_SVM_params = True
    if st.sidebar.checkbox('Выполнять обучение'):
        do_analize_SVM = True
    else:
        do_analize_SVM = False
else:
    show_SVM_params = False

data_X, data_y = preprocess_data(data_no_null)

if do_analize_KNN:
    KNN(data_X, data_y)

if show_SVM_params:
    columns = st.multiselect('Столбцы для построения модели', data_X.columns)
    if do_analize_SVM:
        SVM(data_X[columns], data_y)

# Выводим пользователю сообщение об окончании работы скрипта (на редакции Enterprise multi-session оно полноэкранное)
# Работает только на Windows
if do_analize_KNN or do_analize_SVM:
    if os.name == 'nt' and use_msg:
        cmd_output = subprocess.check_output(['cmd', '/c chcp 65001 & msg * Модель успешно построена!'])

if __name__ == '__main__':
    pass
