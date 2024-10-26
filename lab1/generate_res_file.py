import xlsxwriter
from datetime import datetime


def create_and_right_to_xlsx_file(train_ind, test_ind, valid_ind=None, calibr_ind=None):
    curr_date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fileName = f"split_data_{curr_date_time}.xlsx"

    workbook = xlsxwriter.Workbook(fileName)
    worksheet = workbook.add_worksheet(name='Індекси Підвибірок')

    worksheet.write('A1', 'Навчальна')
    worksheet.write('B1', 'Тестова')
    if valid_ind is not None:
        worksheet.write('C1', 'Валідаційна')
    if calibr_ind is not None:
        worksheet.write('D1', 'Калібрувальна')

    for i in range(0, len(train_ind)):
        worksheet.write(i + 1, 0, train_ind[i])
    for i in range(0, len(test_ind)):
        worksheet.write(i + 1, 1, test_ind[i])
    if valid_ind is not None:
        for i in range(0, len(valid_ind)):
            worksheet.write(i + 1, 2, valid_ind[i])
    if calibr_ind is not None:
        for i in range(0, len(calibr_ind)):
            worksheet.write(i + 1, 3, calibr_ind[i])

    workbook.close()

    return fileName
