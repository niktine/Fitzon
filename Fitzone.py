from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.core.window import Window
from exercise_pose import *


class MyApp(App):
    def build(self):
        # تغییر رنگ پس‌زمینه برنامه
        Window.clearcolor = (1, 1, 1, 1)  # سفید

        # لایه اصلی که شامل عکس و دکمه‌ها است
        main_layout = BoxLayout(orientation='vertical', padding=20, spacing=10)

        # افزودن یک عکس بزرگ در بالای صفحه
        img = Image(source='phooto1.png', size_hint=(1, 0.7), allow_stretch=True)
        main_layout.add_widget(img)

        # لایه دکمه‌ها که از دو بخش چپ و راست تشکیل شده
        button_container = BoxLayout(orientation='horizontal', size_hint=(1, 0.3), spacing=10)

        # بخش دکمه‌های سمت چپ
        left_buttons = BoxLayout(orientation='vertical', spacing=10)
        btn1 = Button(text='dumbbell', background_color=(0.2, 0.6, 1, 1), font_size=20)
        btn2 = Button(text='fly', background_color=(0.2, 0.6, 1, 1), font_size=20)
        btn3 = Button(text='dumbbell lateral raise', background_color=(0.2, 0.6, 1, 1), font_size=20)
        btn4 = Button(text='sit ups', background_color=(0.2, 0.6, 1, 1), font_size=20)

        btn1.bind(on_press=dumbbell_pose)
        btn2.bind(on_press=fly_pose)
        btn3.bind(on_press=dumbbell_lateral_raise_pose)
        btn4.bind(on_press=sit_ups_pose)

        left_buttons.add_widget(btn1)
        left_buttons.add_widget(btn2)
        left_buttons.add_widget(btn3)
        left_buttons.add_widget(btn4)

        # بخش دکمه‌های سمت راست
        right_buttons = BoxLayout(orientation='vertical', spacing=10)
        btn5 = Button(text='lower back', background_color=(1, 0.4, 0.4, 1), font_size=20)
        btn6 = Button(text='leg raise', background_color=(1, 0.4, 0.4, 1), font_size=20)
        btn7 = Button(text='push ups', background_color=(1, 0.4, 0.4, 1), font_size=20)
        btn8 = Button(text='squats_pose()', background_color=(1, 0.4, 0.4, 1), font_size=20)

        btn5.bind(on_press=lower_back_pose)
        btn6.bind(on_press=leg_raise_pose)
        btn7.bind(on_press=push_ups_pose)
        btn8.bind(on_press=squats_pose)

        right_buttons.add_widget(btn5)
        right_buttons.add_widget(btn6)
        right_buttons.add_widget(btn7)
        right_buttons.add_widget(btn8)

        # افزودن دو بخش چپ و راست به لایه اصلی دکمه‌ها
        button_container.add_widget(left_buttons)
        button_container.add_widget(right_buttons)

        # افزودن لایه دکمه‌ها به لایه اصلی
        main_layout.add_widget(button_container)

        return main_layout


if __name__ == '__main__':
    MyApp().run()