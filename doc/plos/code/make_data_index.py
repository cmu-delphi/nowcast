import datetime
import os


def main():
  print('<!doctype html><html><ul>')
  for item in sorted(os.listdir('.')):
    if item[:1] == '.' or item == 'index.html' or item[-3:] == '.py':
      continue
    st = os.stat(item)
    size = st.st_size / (1 << 20)
    fmt = '%Y-%m-%d %H:%M:%S'
    when = datetime.datetime.fromtimestamp(st.st_ctime).strftime(fmt)
    if os.path.isdir(item):
      item += '/'
    link = '<a href="%s">%s</a>' % (item, item)
    print('<li>%s<br>%.2f MiB</li>' % (link, size))
  print('</ul></html>')


if __name__ == '__main__':
  main()
