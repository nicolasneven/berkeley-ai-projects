import bz2, base64

def main():
    with open('submit.token', 'rb') as f:
        token = f.read()
        token = bz2.decompress(token)
        token = base64.b85decode(token)
        token = token.decode('utf-8')

        tokenLines = token.split('%')
        for line in tokenLines:
            print(line)

if __name__ == '__main__':
    main()
