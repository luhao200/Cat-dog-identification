import tkinter
import tkinter.filedialog
import tensorflow as tf
from keras.models import load_model
import numpy as np
from PIL import Image, ImageTk
global patht, pathmod


# 选择并显示图片
def choosepic():
    global patht
    patht = tkinter.filedialog.askopenfilename()
    img_open = Image.open(patht)
    img = ImageTk.PhotoImage(img_open)
    lableShowImage.config(image=img)
    lableShowImage.image = img

# 选择模型
def choosemod():
    global pathmod
    pathmod = tkinter.filedialog.askopenfilename()

# 运行程序
def run():
    model_path = pathmod
    model = load_model(model_path)

    pil_im = Image.open(patht, 'r')
    # 首先更改图片的大小
    pil_im = pil_im.resize((200, 200))
    # 将格式转为numpy array格式
    array_im = np.asarray(pil_im)
    # array_im = array_im.resize((4,4))
    array_im = array_im[np.newaxis, :]
    # 对图像检测
    result = model.predict([[array_im]])
    if result[0][0] > 0.5:
        var.set("狗")
    else:
        var.set("猫")
    tf.compat.v1.reset_default_graph()  # 清除当前默认图中堆栈，重置默认图，实现模型参数的多次读取


if __name__ == '__main__':
    # 生成tk界面 app即主窗口
    app = tkinter.Tk()
    # 修改窗口titile
    app.title("猫狗识别程序")
    # 设置主窗口的大小和位置
    app.geometry("800x400")
    app.config(bg="#FCFEFF")

    # 布局
    top_Frame = tkinter.Frame(app,
                              width=790,
                              height=90,
                              highlightbackground="black",
                              highlightthickness=1,
                              bd=3,
                              bg='#063E7B')
    left_Frame = tkinter.Frame(app,
                               width=190,
                               height=390,
                               highlightbackground="black",
                               highlightthickness=1,
                               bd=3,
                               bg='#FAF5E1')
    main_Frame = tkinter.Frame(app,
                               width=380,
                               height=390,
                               highlightbackground="black",
                               highlightthickness=1,
                               bd=3,
                               bg='#FAF5E1')
    right_Frame = tkinter.Frame(app,
                                width=190,
                                height=390,
                                highlightbackground="black",
                                highlightthickness=1,
                                bd=3,
                                bg='#FAF5E1')
    top_Frame.grid(row=0, column=0, columnspan=3, padx=5, pady=0)
    left_Frame.grid(row=1, column=0, padx=5, pady=5)
    main_Frame.grid(row=1, column=1, padx=0, pady=5)
    right_Frame.grid(row=1, column=2, padx=5, pady=5)
    Selm1 = tkinter.Label(app, text='按钮区', bg='#B3262F')
    Selm1.place(x=50, y=25, height=50, width=100)
    Selm2 = tkinter.Label(app, text='图片显示', bg='#B3262F')
    Selm2.place(x=350, y=25, height=50, width=100)
    Selm3 = tkinter.Label(app, text='说明', bg='#B3262F')
    Selm3.place(x=650, y=25, height=50, width=100)

    # 选择模型的按钮
    buttonSelm = tkinter.Button(app, text='选择模型.h5文件', command=choosemod, bg='#F9FBFC')
    buttonSelm.place(x=50, y=125, height=50, width=100)

    # 选择图片的按钮
    buttonSelImage = tkinter.Button(app, text='选择图片', command=choosepic, bg='#F9FBFC')
    buttonSelImage.place(x=50, y=225, height=50, width=100)

    # 运行的按钮
    buttonSely = tkinter.Button(app, text='运行', command=run, bg='#F9FBFC')
    buttonSely.place(x=50, y=325, height=50, width=100)

    # 图片显示框
    lableShowImage = tkinter.Label(app, bg='#FAF5E1')
    lableShowImage.place(x=225, y=125, height=250, width=350)

    # 说明
    lableS = tkinter.Message(app, text='''第一步，选择训练好的模型。第二步，选择要检测的图片。第三步，运行检测程序。''', bg='#F9FBFC')
    lableS.place(x=625, y=125, height=150, width=150)

    # 结果框
    var = tkinter.StringVar()  # 储存文字的类
    var.set("等待结果")  # 设置文字
    msg = tkinter.Label(app, textvariable=var, relief="sunken", bg='#F9FBFC')
    msg.place(x=650, y=325, height=50, width=100)

    app.mainloop()
