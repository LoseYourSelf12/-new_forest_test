name_id_dict = {'01.01 Прочая реклама': 1, 
                '02.01 Предвыборная Агитация': 2, 
                '02.02 Иная политическая реклама': 3, 
                '02.03 Объявление о работе': 4, 
                '02.04 Объявление о продаже': 5, 
                '02.05 Адвокаты': 6, 
                '05.01 Сравнения (лучший и т.д.)': 7, 
                '05.02 Иностранные слова': 8, 
                '05.03 Информационная продукция': 9, 
                '05.04 Запрещенные информ. ресурсы': 10, 
                '05.05 Физическое лицо': 11, 
                '05.06 Банкротство': 12, 
                '05.07 QR-код / адрес сайта': 13, 
                '08.01 Дистанционные продажи': 14, 
                '09.01 Стимулирующее мероприятие': 15, 
                '09.02 Иные акции': 16, 
                '10.01 Социальная реклама': 17, 
                '21.01 Алкоголь, демонстрация процесса потребления алкоголя': 18, 
                '21.02 Алкомаркет': 19, 
                '21.03 Бар, ресторан': 20, 
                '21.04 Безалкогольное пиво/вино': 21, 
                '24.01 Медицинские услуги': 22, 
                '24.02 Медицинские изделия': 23, 
                '24.03 Лекарственные препараты': 24, 
                '24.04 Методы народной медицины': 25, 
                '24.05 Методы лечения, профилактики и диагностики': 26, 
                '24.06 МедОрганизация/Аптека': 27, 
                '25.01 БАД': 28, 
                '25.02 Детское питание': 29, 
                '27.01 Спорт + букмекер (основанные на риске игры, пари (азартные игры, букмекерские конторы и т.д.))': 30, 
                '27.02 Лотерея': 31, 
                '28.01 Финансовая организация (банк, брокер, страхование и т.д.)': 32, 
                '28.02 Кредит/Ипотека': 33, 
                '28.03 Вклад': 34, 
                '28.04 Доверительное управление': 35, 
                '28.05 Услуги форекс-дилеров': 36, 
                '28.06 Инвест-платформа': 37, 
                '28.07 Строительство (ДДУ)': 38, 
                '28.08 Застройщик': 39, 
                '28.09 Архитектурный проект': 40, 
                '28.10 Построенная недвижимость (продажа/аренда)': 41, 
                '28.11 Земельные участки': 42, 
                '28.12 Рассрочка': 43, 
                '28.13 Строительство (Кооператив)': 44, 
                '28.14 Ломбарды': 45, 
                '29.01 Ценные бумаги': 46, 
                '29.02 Цифровые финансовые активы': 47, 
                '29.03 Криптовалюта': 48, 
                'Оружие и продукция военного назначения': 49, 
                'Нетрадиционные сексуальные отношения и (или) предпочтения': 50, 
                'Педофилия': 51, 
                'Смена пола': 52, 
                'Аудиовизуальный сервис': 53, 
                'ЭТИКА': 54, 
                'Наркотических веществ и их производных': 55, 
                'Взрывчатых веществ и материалов, за исключением пиротехнических изделий': 56, 
                'Органов и (или) тканей человека в качестве объектов купли-продажи': 57, 
                'Товаров, на производство и (или) реализацию которых требуется получение лицензий или иных специальных разрешений, в случае отсутствия таких разрешений': 58, 
                'Табак, табачная продукция, демонстрация процесса курения': 59, 
                'Продукция и бренды автомобильной промышленности': 60, 
                'Автосалоны': 61, 
                'Дилеры личных автомобилей': 62, 
                'Сервисы проката автомобилей (в т.ч. каршеринг)': 63, 
                'Такси-сервисы': 64, 
                'Автострахование': 65, 
                'Лизинг автомобилей': 66, 
                'Продажа/аренда автомобилей': 67, 
                'Автогонки и автомобильные соревнования': 68, 
                'Финансовые пирамиды': 69, 
                'Нижнее белье': 70, 
                'Предметы личной гигиены': 71, 
                'Микрофинансовые организации': 72, 
                'Коллекторские агентства': 73, 
                'Обучающие курсы, тренинги, книги, авторами которых являются люди, не обладающие специальной квалификацией': 74, 
                'Ритуальные услуги': 75, 
                'Энергетические напитки': 76, 
                'Наименования, флаги и символику недружественных государств': 77, 
                'Страницы в социальных сетях физических лиц, в т.ч. телеграмм-каналы': 78, 
                'Обнаженные тела': 79, 
                'Негативный посыл по отношению к транспорту (например, запрыгнуть в последний вагон)': 80, 
                'Страшные, пугающие изображения': 81, 
                'Образы смерти': 82, 
                'Казино (в т.ч. онлайн-казино)': 83, 
                'Иностранные социальные сети': 84, 
                'Дискредитация СВО и ее участников, а также традиций, истории, граждан РФ, самой РФ и органов власти РФ': 85, 
                'Торговые и развлекательные центры ': 86, 
                'Сервисы, предоставляющие услуги доставки еды или готовых блюд': 87, 
                'Гостиницы и прочие места для временного проживания': 88, 
                'Сервисы, предоставляющие услуги по бронированию и иные туристские услуги': 89, 
                'Религиозные объединения': 90}