import bz2, base64

s = """
import autograder
import projectParams
import re
import os
import random
from datetime import datetime
import bz2, base64

def main():
    codePaths = projectParams.STUDENT_CODE_DEFAULT.split(',')
    # moduleCodeDict = {}
    # for cp in codePaths:
    #     moduleName = re.match('.*?([^/]*)\.py', cp).group(1)
    #     moduleCodeDict[moduleName] = readFile(cp, root=options.codeRoot)
    # moduleCodeDict['projectTestClasses'] = readFile(options.testCaseCode, root=options.codeRoot)
    # moduleDict = loadModuleDict(moduleCodeDict)

    moduleDict = {}
    for cp in codePaths:
        moduleName = re.match('.*?([^/]*)\.py', cp).group(1)
        moduleDict[moduleName] = autograder.loadModuleFile(moduleName, os.path.join('', cp))
    moduleName = re.match('.*?([^/]*)\.py', projectParams.PROJECT_TEST_CLASSES).group(1)
    moduleDict['projectTestClasses'] = autograder.loadModuleFile(moduleName, os.path.join('', projectParams.PROJECT_TEST_CLASSES))

    random.seed(datetime.now())
    rand = random.randint(42,424242)

    points = autograder.evaluate(False,'test_cases',moduleDict,forSubmission=True)
    total = sum(points.values())
    token = '%'.join([str(rand),str(points),str(total)])
    token = bytes(token, 'utf-8')
    token = base64.b85encode(token)
    token = bz2.compress(token)
    with open('submit.token', 'wb') as f:
        f.write(token)

if __name__ == '__main__':
    main()
"""

s = bytes(s,'utf-8')
s = bz2.compress(s)
s = base64.b64encode(s)

with open('encodedSubmit.txt', 'wb') as f:
    f.write(s)
