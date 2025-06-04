import base
import ocr
import yolo
import mixer

# нет категории: 01.01 Прочая реклама

base.Category(
    name="02.01 Предвыборная Агитация",
    detectors=[yolo.Kat_02_01()],
    mixer=mixer.SimpleMax()
)


base.Category(
    name="02.02 Иная политическая реклама",
    detectors=[yolo.Kat_02_02()],
    mixer=mixer.SimpleMax()
)


base.Category(
    name="02.03 Объявление о работе",
    detectors=[ocr.Kat_02_03()],
    mixer=mixer.SimpleMax()
)


base.Category(
    name="02.04 Объявление о продаже",
    detectors=[ocr.Kat_02_04()],
    mixer=mixer.SimpleMax()
)


base.Category(
    name="02.05 Адвокаты",
    detectors=[yolo.Kat_02_05(), ocr.Kat_02_05()],
    mixer=mixer.RFRScikit('mix_02_05')
)


base.Category(
    name="05.01 Сравнения (лучший и т.д.)",
    detectors=[ocr.Kat_05_01()],
    mixer=mixer.SimpleMax()
)

# не сделанная категория: 05.02 Иностранные слова


base.Category(
    name="05.03 Информационная продукция",
    detectors=[yolo.Kat_05_03(), ocr.Kat_05_03()],
    mixer=mixer.RFRScikit('mix_05_03')
)


base.Category(
    name="05.04 Запрещенные информ. ресурсы",
    detectors=[yolo.Kat_05_04()],
    mixer=mixer.SimpleMax()
)


base.Category(
    name="05.05 Физическое лицо",
    detectors=[yolo.Kat_05_05()],
    mixer=mixer.SimpleMax()
)


base.Category(
    name="05.06 Банкротство",
    detectors=[ocr.Kat_05_06()],
    mixer=mixer.SimpleMax()
)


base.Category(
    name="05.07 QR-код / адрес сайта",
    detectors=[yolo.Kat_05_07(), ocr.Kat_05_07()],
    mixer=mixer.RFRScikit('mix_05_07')
)


base.Category(
    name="08.01 Дистанционные продажи",
    detectors=[yolo.Kat_08_01()],
    mixer=mixer.RFRScikit('mix_08_01')
)


base.Category(
    name="09.01 Стимулирующее мероприятие",
    detectors=[ocr.Kat_09_01()],
    mixer=mixer.SimpleMax()
)


base.Category(
    name="09.02 Иные акции",
    detectors=[yolo.Kat_09_02(), ocr.Kat_09_02()],
    mixer=mixer.RFRScikit('mix_09_02')
)


base.Category(
    name="10.01 Социальная реклама",
    detectors=[yolo.Kat_10_01(), ocr.Kat_10_01()],
    mixer=mixer.RFRScikit('mix_10_01')
)


base.Category(
    name="21.01 Алкоголь, демонстрация процесса потребления алкоголя",
    detectors=[yolo.Kat_21_01()],
    mixer=mixer.SimpleMax()
)


base.Category(
    name="21.02 Алкомаркет",
    detectors=[yolo.Kat_21_02()],
    mixer=mixer.SimpleMax()
)


base.Category(
    name="21.03 Бар, ресторан",
    detectors=[yolo.Kat_21_03(), ocr.Kat_21_03()],
    mixer=mixer.SimpleMax()
)


base.Category(
    name="21.04 Безалкогольное пиво/вино",
    detectors=[yolo.Kat_21_04()],
    mixer=mixer.SimpleMax()
)


base.Category(
    name="24.01 Медицинские услуги",
    detectors=[yolo.Kat_24_01()],
    mixer=mixer.SimpleMax()
)


base.Category(
    name="24.02 Медицинские изделия",
    detectors=[yolo.Kat_24_02(), ocr.Kat_24_02()],
    mixer=mixer.RFRScikit('mix_24_02')
)


base.Category(
    name="24.03 Лекарственные препараты",
    detectors=[yolo.Kat_24_03(), ocr.Kat_24_03()],
    mixer=mixer.RFRScikit('mix_24_03')
)


base.Category(
    name="24.04 Методы народной медицины",
    detectors=[yolo.Kat_24_04()],
    mixer=mixer.SimpleMax()
)


base.Category(
    name="24.05 Методы лечения, профилактики и диагностики",
    detectors=[yolo.Kat_24_05()],
    mixer=mixer.SimpleMax()
)


base.Category(
    name="24.06 МедОрганизация/Аптека",
    detectors=[yolo.Kat_24_06(), ocr.Kat_24_06()],
    mixer=mixer.RFRScikit('mix_24_06')
)

base.Category(
    name="25.01 БАД",
    detectors=[ocr.Kat_25_01()],
    mixer=mixer.SimpleMax()
)


base.Category(
    name="25.02 Детское питание",
    detectors=[yolo.Kat_25_02()],
    mixer=mixer.SimpleMax()
)


base.Category(
    name="27.01 Спорт + букмекер (основанные на риске игры, пари (азартные игры, букмекерские конторы и т.д.))",
    detectors=[yolo.Kat_27_01()],
    mixer=mixer.SimpleMax()
)


base.Category(
    name="27.02 Лотерея",
    detectors=[ocr.Kat_27_02()],
    mixer=mixer.SimpleMax()
)


base.Category(
    name="28.01 Финансовая организация (банк, брокер, страхование и т.д.)",
    detectors=[ocr.Kat_28_01()],
    mixer=mixer.SimpleMax()
)


base.Category(
    name="28.02 Кредит/Ипотека",
    detectors=[yolo.Kat_28_02()],
    mixer=mixer.SimpleMax()
)


base.Category(
    name="28.03 Вклад",
    detectors=[ocr.Kat_28_03()],
    mixer=mixer.SimpleMax()
)


base.Category(
    name="28.04 Доверительное управление",
    detectors=[ocr.Kat_28_04()],
    mixer=mixer.SimpleMax()
)


base.Category(
    name="28.05 Услуги форекс-дилеров",
    detectors=[ ocr.Kat_28_05()],
    mixer=mixer.SimpleMax()
)


base.Category(
    name="28.06 Инвест-платформа",
    detectors=[yolo.Kat_28_06(), ocr.Kat_28_06()],
    mixer=mixer.SimpleMax()
)


base.Category(
    name="28.07 Строительство (ДДУ)",
    detectors=[yolo.Kat_28_07()],
    mixer=mixer.SimpleMax()
)


base.Category(
    name="28.08 Застройщик",
    detectors=[yolo.Kat_28_08(), ocr.Kat_28_08()],
    mixer=mixer.RFRScikit('mix_28_08')
)


base.Category(
    name="28.09 Архитектурный проект",
    detectors=[ocr.Kat_28_09()],
    mixer=mixer.SimpleMax()
)


base.Category(
    name="28.10 Построенная недвижимость (продажа/аренда)",
    detectors=[ocr.Kat_28_10()],
    mixer=mixer.SimpleMax()
)


base.Category(
    name="28.11 Земельные участки",
    detectors=[yolo.Kat_28_11(), ocr.Kat_28_11()],
    mixer=mixer.RFRScikit('mix_28_11')
)


base.Category(
    name="28.12 Рассрочка",
    detectors=[yolo.Kat_28_12()],
    mixer=mixer.SimpleMax()
)


base.Category(
    name="28.13 Строительство (Кооператив)",
    detectors=[ocr.Kat_28_13()],
    mixer=mixer.SimpleMax()
)


base.Category(
    name="28.14 Ломбарды",
    detectors=[yolo.Kat_28_14(), ocr.Kat_28_14()],
    mixer=mixer.RFRScikit('mix_28_14')
)


base.Category(
    name="29.01 Ценные бумаги",
    detectors=[yolo.Kat_29_01()],
    mixer=mixer.SimpleMax()
)


base.Category(
    name="29.02 Цифровые финансовые активы",
    detectors=[yolo.Kat_29_02()],
    mixer=mixer.SimpleMax()
)


base.Category(
    name="29.03 Криптовалюта",
    detectors=[yolo.Kat_29_03()],
    mixer=mixer.SimpleMax()
)
