

#include <qcoreapplication.h>
#include <qdiriterator.h>
#include <qfile.h>
#include <qmetaobject.h>
#include <qstring.h>
#include <qtextstream.h>
#include <qcommandlineparser.h>

#include <qdebug.h>

#include <limits.h>
#include <stdio.h>

static bool printFilename = true;
static bool modify = false;

QString signature(const QString &line, int pos)
{
    int start = pos;
    
    while (start < line.length() && line.at(start) != QLatin1Char('('))
        ++start;
    int i = ++start;
    int par = 1;
    
    while (i < line.length() && par > 0) {
        if (line.at(i) == QLatin1Char('('))
            ++par;
        else if (line.at(i) == QLatin1Char(')'))
            --par;
        ++i;
    }
    if (par == 0)
        return line.mid(start, i - start - 1);
    return QString();
}

bool checkSignature(const QString &fileName, QString &line, const char *sig)
{
    static QStringList fileList;

    int idx = -1;
    bool found = false;
    while ((idx = line.indexOf(sig, ++idx)) != -1) {
        const QByteArray sl(signature(line, idx).toLocal8Bit());
        QByteArray nsl(QMetaObject::normalizedSignature(sl.constData()));
        if (sl != nsl) {
            found = true;
            if (printFilename && !fileList.contains(fileName)) {
                fileList.prepend(fileName);
                printf("%s\n", fileName.toLocal8Bit().constData());
            }
            if (modify)
                line.replace(sl, nsl);
            
        }
    }
    return found;
}

void writeChanges(const QString &fileName, const QStringList &lines)
{
    QFile file(fileName);
    if (!file.open(QIODevice::WriteOnly)) {
        qDebug("unable to open file '%s' for writing (%s)", fileName.toLocal8Bit().constData(), file.errorString().toLocal8Bit().constData());
        return;
    }
    QTextStream stream(&file);
    for (int i = 0; i < lines.count(); ++i)
        stream << lines.at(i);
    file.close();
}

void check(const QString &fileName)
{
    QFile file(fileName);
    if (!file.open(QIODevice::ReadOnly)) {
        qDebug("unable to open file: '%s' (%s)", fileName.toLocal8Bit().constData(), file.errorString().toLocal8Bit().constData());
        return;
    }
    QStringList lines;
    bool found = false;
    while (true) {
        QByteArray bline = file.readLine(16384);
        if (bline.isEmpty())
            break;
        QString line = QString::fromLocal8Bit(bline);
        Q_ASSERT_X(line.endsWith("\n"), "check()", fileName.toLocal8Bit().constData());
        found |= checkSignature(fileName, line, "SLOT");
        found |= checkSignature(fileName, line, "SIGNAL");
        if (modify)
            lines << line;
    }
    file.close();

    if (found && modify) {
        printf("Modifying file: '%s'\n", fileName.toLocal8Bit().constData());
        writeChanges(fileName, lines);
    }
}

void traverse(const QString &path)
{
    QDirIterator dirIterator(path, QDir::NoDotAndDotDot | QDir::Dirs | QDir::Files | QDir::NoSymLinks);

    while (dirIterator.hasNext()) {
        QString filePath = dirIterator.next();
        if (filePath.endsWith(".cpp"))
            check(filePath);
        else if (QFileInfo(filePath).isDir())
            traverse(filePath); 
    }
}

int main(int argc, char *argv[])
{
    QCoreApplication app(argc, argv);

    QCommandLineParser parser;
    parser.setApplicationDescription(
            QStringLiteral("Qt Normalize tool (Qt %1)\nOutputs all filenames that contain non-normalized SIGNALs and SLOTs")
            .arg(QString::fromLatin1(QT_VERSION_STR)));
    parser.addHelpOption();
    parser.addVersionOption();

    QCommandLineOption modifyOption(QStringLiteral("modify"),
                                    QStringLiteral("Fix all occurrences of non-normalized SIGNALs and SLOTs."));
    parser.addOption(modifyOption);

    parser.addPositionalArgument(QStringLiteral("path"),
                                 QStringLiteral("can be a single file or a directory (in which case, look for *.cpp recursively)"));

    parser.process(app);

    if (parser.positionalArguments().count() != 1)
        parser.showHelp(1);
    QString path = parser.positionalArguments().first();
    if (path == "-")
        parser.showHelp(1);

    if (parser.isSet(modifyOption)) {
        printFilename = false;
        modify = true;
    }

    QFileInfo fi(path);
    if (fi.isFile()) {
        check(path);
    } else if (fi.isDir()) {
        if (!path.endsWith("/"))
            path.append("/");
        traverse(path);
    } else {
        qWarning("Don't know what to do with '%s'", path.toLocal8Bit().constData());
        return 1;
    }

    return 0;
}
