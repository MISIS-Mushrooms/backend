from geopy import Nominatim

if __name__ == '__main__':
    locator = Nominatim(user_agent='Mozilla/5.0 (Windows NT 10.0; WOW64; rv:45.0) Gecko/20100101 Firefox/45.0')
    res = locator.geocode('город Москва, Саратовская улица, дом 16, корпус 2')
    print(123)