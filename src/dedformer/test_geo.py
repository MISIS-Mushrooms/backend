from geopy import Nominatim

from dedformer.model import UserBank, GroupBank

if __name__ == '__main__':
    users = UserBank('data/users.csv')
    groups = GroupBank('data/groups.csv')
    print(123)