import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Пресеты параметров различных жидкостей
liquids = {
    'Вода': {'K': 0.001, 'n': 1.0},
    'Мёд': {'K': 10.0, 'n': 0.4},
    'Моторное масло': {'K': 0.2, 'n': 0.8},
    'Шампунь': {'K': 3.0, 'n': 0.6}
}

def calculate_viscosity(K, n, shear_rate):
    return K * shear_rate**(n - 1)

def smooth_data(data, window_size):
    """Функция для сглаживания данных с использованием скользящего среднего."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

def generate_graph(liquid_name, low_speed, high_speed, duration, steps, smooth_weight=0.5):
    if liquid_name not in liquids:
        st.error(f"Жидкость '{liquid_name}' не найдена в пресетах.")
        return

    K = liquids[liquid_name]['K']
    n = liquids[liquid_name]['n']

    time_points = np.linspace(0, duration, num=duration*10)  # временные точки каждые 0.1 секунды
    step_duration = duration / steps
    speed = np.zeros_like(time_points)
    viscosity = np.zeros_like(time_points)
    viscosity_s = np.zeros_like(time_points)

    for i, t in enumerate(time_points):
        current_step = int(t // step_duration)
        next_step = current_step + 1
        if next_step >= steps:
            next_step = steps - 1
        current_speed = low_speed + (high_speed - low_speed) * current_step / steps
        next_speed = low_speed + (high_speed - low_speed) * next_step / steps

        # Сглаживание на переходах ступеней
        transition_ratio = (t % step_duration) / step_duration
        interpolated_speed = current_speed + (next_speed - current_speed) * transition_ratio

        speed[i] = current_speed
        # Добавление небольшого случайного шума для сглаживания функции вязкости
        viscosity_value = calculate_viscosity(K, n, current_speed)
        viscosity[i] = viscosity_value #+ np.random.normal(0, viscosity_value )/ 100
        viscosity_value = calculate_viscosity(K, n, interpolated_speed)
        viscosity_s[i] = viscosity_value# + np.random.normal(0, viscosity_value )/ 100

    # Сглаживание данных вязкости с меньшим окном
    smoothed_viscosity = smooth_data(viscosity_s, window_size=3)

    # Взвешенное смешивание исходной и сглаженной вязкости
    mixed_viscosity = (1 - smooth_weight) * viscosity + smooth_weight * smoothed_viscosity

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Время (с)')
    ax1.set_ylabel('Вязкость (Па·с)', color='tab:blue')
    ax1.plot(time_points, mixed_viscosity, label='Вязкость', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Скорость вращения (об/мин)', color='tab:green')
    ax2.plot(time_points, speed, label='Скорость вращения', color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    fig.tight_layout()
    st.pyplot(fig)

# Интерфейс Streamlit
st.title('График вязкости и скорости вращения')
liquid_name = st.selectbox('Выберите жидкость', list(liquids.keys()))
low_speed = st.slider('Минимальная скорость вращения (об/мин)', 1, 100, 5)
high_speed = st.slider('Максимальная скорость вращения (об/мин)', 10, 1000, 50)
duration = st.slider('Длительность эксперимента (с)', 10, 1000, 300)
steps = st.slider('Количество ступеней', 2, 50, 20)
smooth_weight = st.slider('Весовой коэффициент сглаживания', 0.0, 1.0, 0.1)

generate_graph(liquid_name, low_speed, high_speed, duration, steps, smooth_weight)
