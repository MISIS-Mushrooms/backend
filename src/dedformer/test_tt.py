import datetime

from dedformer.model import GroupBank

if __name__ == '__main__':
    bank = GroupBank('data/groups.csv')
    print(bank.get_timetable('801370407', datetime.date(2023, 5, 1)))