/****************************************************************************
**
** Copyright (C) 2012 Nokia Corporation and/or its subsidiary(-ies).
** Contact: http://www.qt-project.org/
**
** This file is part of the qmake application of the Qt Toolkit.
**
** $QT_BEGIN_LICENSE:LGPL$
** GNU Lesser General Public License Usage
** This file may be used under the terms of the GNU Lesser General Public
** License version 2.1 as published by the Free Software Foundation and
** appearing in the file LICENSE.LGPL included in the packaging of this
** file. Please review the following information to ensure the GNU Lesser
** General Public License version 2.1 requirements will be met:
** http://www.gnu.org/licenses/old-licenses/lgpl-2.1.html.
**
** In addition, as a special exception, Nokia gives you certain additional
** rights. These rights are described in the Nokia Qt LGPL Exception
** version 1.1, included in the file LGPL_EXCEPTION.txt in this package.
**
** GNU General Public License Usage
** Alternatively, this file may be used under the terms of the GNU General
** Public License version 3.0 as published by the Free Software Foundation
** and appearing in the file LICENSE.GPL included in the packaging of this
** file. Please review the following information to ensure the GNU General
** Public License version 3.0 requirements will be met:
** http://www.gnu.org/copyleft/gpl.html.
**
** Other Usage
** Alternatively, this file may be used in accordance with the terms and
** conditions contained in a signed written agreement between you and Nokia.
**
**
**
**
**
**
** $QT_END_LICENSE$
**
****************************************************************************/

#include "meta.h"
#include "option.h"
#include "project.h"
#include "winmakefile.h"
#include <qdir.h>
#include <qhash.h>
#include <qregexp.h>
#include <qstring.h>
#include <qstringlist.h>
#include <qtextstream.h>
#include <stdlib.h>

QT_BEGIN_NAMESPACE

Win32MakefileGenerator::Win32MakefileGenerator() : MakefileGenerator() {}

int Win32MakefileGenerator::findHighestVersion(const QString &d,
                                               const QString &stem,
                                               const QString &ext) {
    QString bd = Option::fixPathToLocalOS(d, true);
    if (!exists(bd))
        return -1;

    QMakeMetaInfo libinfo;
    bool libInfoRead = libinfo.readLib(bd + Option::dir_sep + stem);

    // If the library, for which we're trying to find the highest version
    // number, is a static library
    if (libInfoRead && libinfo.values("QMAKE_PRL_CONFIG").contains("staticlib"))
        return -1;

    if (!project->values("QMAKE_" + stem.toUpper() + "_VERSION_OVERRIDE")
             .isEmpty())
        return project->values("QMAKE_" + stem.toUpper() + "_VERSION_OVERRIDE")
            .first()
            .toInt();

    int biggest = -1;
    if (!project->isActiveConfig("no_versionlink")) {
        static QHash<QString, QStringList> dirEntryListCache;
        QStringList entries = dirEntryListCache.value(bd);
        if (entries.isEmpty()) {
            QDir dir(bd);
            entries = dir.entryList();
            dirEntryListCache.insert(bd, entries);
        }

        QRegExp regx(QString("((lib)?%1([0-9]*)).(%2|prl)$").arg(stem).arg(ext),
                     Qt::CaseInsensitive);
        for (QStringList::Iterator it = entries.begin(); it != entries.end();
             ++it) {
            if (regx.exactMatch((*it))) {
                if (!regx.cap(3).isEmpty()) {
                    bool ok = true;
                    int num = regx.cap(3).toInt(&ok);
                    biggest = qMax(biggest, (!ok ? -1 : num));
                }
            }
        }
    }
    if (libInfoRead &&
        !libinfo.values("QMAKE_PRL_CONFIG").contains("staticlib") &&
        !libinfo.isEmpty("QMAKE_PRL_VERSION"))
        biggest =
            qMax(biggest,
                 libinfo.first("QMAKE_PRL_VERSION").replace(".", "").toInt());
    return biggest;
}

bool Win32MakefileGenerator::findLibraries() {
    QList<QMakeLocalFileName> dirs;
    const QString lflags[] = {"QMAKE_LIBS", "QMAKE_LIBS_PRIVATE", QString()};
    for (int i = 0; !lflags[i].isNull(); i++) {
        QStringList &l = project->values(lflags[i]);
        for (QStringList::Iterator it = l.begin(); it != l.end();) {
            QChar quote;
            bool modified_opt = false, remove = false;
            QString opt = (*it).trimmed();
            if ((opt[0] == '\'' || opt[0] == '"') &&
                opt[(int)opt.length() - 1] == opt[0]) {
                quote = opt[0];
                opt = opt.mid(1, opt.length() - 2);
            }
            if (opt.startsWith("/LIBPATH:")) {
                dirs.append(QMakeLocalFileName(opt.mid(9)));
            } else if (opt.startsWith("-L") || opt.startsWith("/L")) {
                QString libpath = opt.mid(2);
                QMakeLocalFileName l(libpath);
                if (!dirs.contains(l)) {
                    dirs.append(l);
                    modified_opt = true;
                    if (!quote.isNull()) {
                        libpath = quote + libpath + quote;
                        quote = QChar();
                    }
                    (*it) = "/LIBPATH:" + libpath;
                } else {
                    remove = true;
                }
            } else if (opt.startsWith("-l") || opt.startsWith("/l")) {
                QString lib = opt.right(opt.length() - 2), out;
                if (!lib.isEmpty()) {
                    QString suffix;
                    if (!project->isEmpty("QMAKE_" + lib.toUpper() + "_SUFFIX"))
                        suffix = project->first("QMAKE_" + lib.toUpper() +
                                                "_SUFFIX");
                    for (QList<QMakeLocalFileName>::Iterator it = dirs.begin();
                         it != dirs.end(); ++it) {
                        QString extension;
                        int ver = findHighestVersion((*it).local(), lib);
                        if (ver > 0)
                            extension += QString::number(ver);
                        extension += suffix;
                        extension += ".lib";
                        if (QMakeMetaInfo::libExists((*it).local() +
                                                     Option::dir_sep + lib) ||
                            exists((*it).local() + Option::dir_sep + lib +
                                   extension)) {
                            out = (*it).real() + Option::dir_sep + lib +
                                  extension;
                            if (out.contains(QLatin1Char(' '))) {
                                out.prepend(QLatin1Char('\"'));
                                out.append(QLatin1Char('\"'));
                            }
                            break;
                        }
                    }
                }
                if (out.isEmpty())
                    out = lib + ".lib";
                modified_opt = true;
                (*it) = out;
            } else if (!exists(Option::fixPathToLocalOS(opt))) {
                QList<QMakeLocalFileName> lib_dirs;
                QString file = opt;
                int slsh = file.lastIndexOf(Option::dir_sep);
                if (slsh != -1) {
                    lib_dirs.append(QMakeLocalFileName(file.left(slsh + 1)));
                    file = file.right(file.length() - slsh - 1);
                } else {
                    lib_dirs = dirs;
                }
                if (file.endsWith(".lib")) {
                    file = file.left(file.length() - 4);
                    if (!file.at(file.length() - 1).isNumber()) {
                        QString suffix;
                        if (!project->isEmpty(
                                "QMAKE_" +
                                file.section(Option::dir_sep, -1).toUpper() +
                                "_SUFFIX"))
                            suffix = project->first(
                                "QMAKE_" +
                                file.section(Option::dir_sep, -1).toUpper() +
                                "_SUFFIX");
                        for (QList<QMakeLocalFileName>::Iterator dep_it =
                                 lib_dirs.begin();
                             dep_it != lib_dirs.end(); ++dep_it) {
                            QString lib_tmpl(file + "%1" + suffix + ".lib");
                            int ver =
                                findHighestVersion((*dep_it).local(), file);
                            if (ver != -1) {
                                if (ver)
                                    lib_tmpl = lib_tmpl.arg(ver);
                                else
                                    lib_tmpl = lib_tmpl.arg("");
                                if (slsh != -1) {
                                    QString dir = (*dep_it).real();
                                    if (!dir.endsWith(Option::dir_sep))
                                        dir += Option::dir_sep;
                                    lib_tmpl.prepend(dir);
                                }
                                modified_opt = true;
                                (*it) = lib_tmpl;
                                break;
                            }
                        }
                    }
                }
            }
            if (remove) {
                it = l.erase(it);
            } else {
                if (!quote.isNull() && modified_opt)
                    (*it) = quote + (*it) + quote;
                ++it;
            }
        }
    }
    return true;
}

void Win32MakefileGenerator::processPrlFiles() {
    const QString libArg = project->first("QMAKE_L_FLAG");
    QHash<QString, bool> processed;
    QList<QMakeLocalFileName> libdirs;
    for (bool ret = false; true; ret = false) {
        // read in any prl files included..
        QStringList l_out;
        QStringList l = project->values("QMAKE_LIBS");
        for (QStringList::Iterator it = l.begin(); it != l.end(); ++it) {
            QString opt = (*it).trimmed();
            if ((opt[0] == '\'' || opt[0] == '"') &&
                opt[(int)opt.length() - 1] == opt[0])
                opt = opt.mid(1, opt.length() - 2);
            if (opt.startsWith(libArg)) {
                QMakeLocalFileName l(opt.mid(libArg.length()));
                if (!libdirs.contains(l))
                    libdirs.append(l);
            } else if (!opt.startsWith("/") && !processed.contains(opt)) {
                if (processPrlFile(opt)) {
                    processed.insert(opt, true);
                    ret = true;
                } else if (QDir::isRelativePath(opt) || opt.startsWith("-l")) {
                    QString tmp;
                    if (opt.startsWith("-l"))
                        tmp = opt.mid(2);
                    else
                        tmp = opt;
                    for (QList<QMakeLocalFileName>::Iterator it =
                             libdirs.begin();
                         it != libdirs.end(); ++it) {
                        QString prl = (*it).local() + Option::dir_sep + tmp;
                        // the original is used as the key
                        QString orgprl = prl;
                        if (processed.contains(prl)) {
                            break;
                        } else if (processPrlFile(prl)) {
                            processed.insert(orgprl, true);
                            ret = true;
                            break;
                        }
                    }
                }
            }
            if (!opt.isEmpty())
                l_out.append(opt);
        }
        if (ret)
            l = l_out;
        else
            break;
    }
}

void Win32MakefileGenerator::processVars() {
    // If the TARGET looks like a path split it into DESTDIR and the resulting
    // TARGET
    if (!project->isEmpty("TARGET")) {
        QString targ = project->first("TARGET");
        int slsh =
            qMax(targ.lastIndexOf('/'), targ.lastIndexOf(Option::dir_sep));
        if (slsh != -1) {
            if (project->isEmpty("DESTDIR"))
                project->values("DESTDIR").append("");
            else if (project->first("DESTDIR").right(1) != Option::dir_sep)
                project->values("DESTDIR") =
                    QStringList(project->first("DESTDIR") + Option::dir_sep);
            project->values("DESTDIR") =
                QStringList(project->first("DESTDIR") + targ.left(slsh + 1));
            project->values("TARGET") = QStringList(targ.mid(slsh + 1));
        }
    }

    project->values("QMAKE_ORIG_TARGET") = project->values("TARGET");
    if (project->isEmpty("QMAKE_PROJECT_NAME"))
        project->values("QMAKE_PROJECT_NAME") =
            project->values("QMAKE_ORIG_TARGET");
    else if (project->first("TEMPLATE").startsWith("vc"))
        project->values("MAKEFILE") = project->values("QMAKE_PROJECT_NAME");

    if (!project->values("QMAKE_INCDIR").isEmpty())
        project->values("INCLUDEPATH") += project->values("QMAKE_INCDIR");

    if (!project->values("VERSION").isEmpty()) {
        QStringList l = project->first("VERSION").split('.');
        if (l.size() > 0)
            project->values("VER_MAJ").append(l[0]);
        if (l.size() > 1)
            project->values("VER_MIN").append(l[1]);
    }

    // TARGET_VERSION_EXT will be used to add a version number onto the target
    // name
    if (project->values("TARGET_VERSION_EXT").isEmpty() &&
        !project->values("VER_MAJ").isEmpty())
        project->values("TARGET_VERSION_EXT")
            .append(project->values("VER_MAJ").first());

    if (project->isEmpty("QMAKE_COPY_FILE"))
        project->values("QMAKE_COPY_FILE").append("$(COPY)");
    if (project->isEmpty("QMAKE_COPY_DIR"))
        project->values("QMAKE_COPY_DIR").append("xcopy /s /q /y /i");
    if (project->isEmpty("QMAKE_INSTALL_FILE"))
        project->values("QMAKE_INSTALL_FILE").append("$(COPY_FILE)");
    if (project->isEmpty("QMAKE_INSTALL_PROGRAM"))
        project->values("QMAKE_INSTALL_PROGRAM").append("$(COPY_FILE)");
    if (project->isEmpty("QMAKE_INSTALL_DIR"))
        project->values("QMAKE_INSTALL_DIR").append("$(COPY_DIR)");

    fixTargetExt();
    processRcFileVar();
    processFileTagsVar();

    QStringList &incDir = project->values("INCLUDEPATH");
    for (QStringList::Iterator incDir_it = incDir.begin();
         incDir_it != incDir.end(); ++incDir_it) {
        if (!(*incDir_it).isEmpty())
            (*incDir_it) =
                Option::fixPathToTargetOS((*incDir_it), false, false);
    }

    QString libArg = project->first("QMAKE_L_FLAG");
    QStringList libs;
    QStringList &libDir = project->values("QMAKE_LIBDIR");
    for (QStringList::Iterator libDir_it = libDir.begin();
         libDir_it != libDir.end(); ++libDir_it) {
        if (!(*libDir_it).isEmpty()) {
            (*libDir_it).remove("\"");
            if ((*libDir_it).endsWith("\\"))
                (*libDir_it).chop(1);
            libs << libArg + escapeFilePath(Option::fixPathToTargetOS(
                                 (*libDir_it), false, false));
        }
    }
    project->values("QMAKE_LIBS") +=
        libs + escapeFilePaths(project->values("LIBS"));
    project->values("QMAKE_LIBS_PRIVATE") +=
        escapeFilePaths(project->values("LIBS_PRIVATE"));

    if (project->values("TEMPLATE").contains("app")) {
        project->values("QMAKE_CFLAGS") += project->values("QMAKE_CFLAGS_APP");
        project->values("QMAKE_CXXFLAGS") +=
            project->values("QMAKE_CXXFLAGS_APP");
        project->values("QMAKE_LFLAGS") += project->values("QMAKE_LFLAGS_APP");
    } else if (project->values("TEMPLATE").contains("lib") &&
               project->isActiveConfig("dll")) {
        if (!project->isActiveConfig("plugin") ||
            !project->isActiveConfig("plugin_no_share_shlib_cflags")) {
            project->values("QMAKE_CFLAGS") +=
                project->values("QMAKE_CFLAGS_SHLIB");
            project->values("QMAKE_CXXFLAGS") +=
                project->values("QMAKE_CXXFLAGS_SHLIB");
        }
        if (project->isActiveConfig("plugin")) {
            project->values("QMAKE_CFLAGS") +=
                project->values("QMAKE_CFLAGS_PLUGIN");
            project->values("QMAKE_CXXFLAGS") +=
                project->values("QMAKE_CXXFLAGS_PLUGIN");
            project->values("QMAKE_LFLAGS") +=
                project->values("QMAKE_LFLAGS_PLUGIN");
        } else {
            project->values("QMAKE_LFLAGS") +=
                project->values("QMAKE_LFLAGS_SHLIB");
        }
    }
}

void Win32MakefileGenerator::fixTargetExt() {
    if (project->isEmpty("QMAKE_EXTENSION_STATICLIB"))
        project->values("QMAKE_EXTENSION_STATICLIB").append("lib");
    if (project->isEmpty("QMAKE_EXTENSION_SHLIB"))
        project->values("QMAKE_EXTENSION_SHLIB").append("dll");

    if (!project->values("QMAKE_APP_FLAG").isEmpty()) {
        project->values("TARGET_EXT").append(".exe");
    } else if (project->isActiveConfig("shared")) {
        project->values("TARGET_EXT")
            .append(project->first("TARGET_VERSION_EXT") + "." +
                    project->first("QMAKE_EXTENSION_SHLIB"));
        project->values("TARGET").first() =
            project->first("QMAKE_PREFIX_SHLIB") + project->first("TARGET");
    } else {
        project->values("TARGET_EXT")
            .append("." + project->first("QMAKE_EXTENSION_STATICLIB"));
        project->values("TARGET").first() =
            project->first("QMAKE_PREFIX_STATICLIB") + project->first("TARGET");
    }
}

void Win32MakefileGenerator::processRcFileVar() {
    if (Option::qmake_mode == Option::QMAKE_GENERATE_NOTHING)
        return;

    if (((!project->values("VERSION").isEmpty()) &&
         project->values("RC_FILE").isEmpty() &&
         project->values("RES_FILE").isEmpty() &&
         !project->isActiveConfig("no_generated_target_info") &&
         (project->isActiveConfig("shared") ||
          !project->values("QMAKE_APP_FLAG").isEmpty())) ||
        !project->values("QMAKE_WRITE_DEFAULT_RC").isEmpty()) {

        QByteArray rcString;
        QTextStream ts(&rcString, QFile::WriteOnly);

        QStringList vers = project->values("VERSION").first().split(".");
        for (int i = vers.size(); i < 4; i++)
            vers += "0";
        QString versionString = vers.join(".");

        QString companyName;
        if (!project->values("QMAKE_TARGET_COMPANY").isEmpty())
            companyName = project->values("QMAKE_TARGET_COMPANY").join(" ");

        QString description;
        if (!project->values("QMAKE_TARGET_DESCRIPTION").isEmpty())
            description = project->values("QMAKE_TARGET_DESCRIPTION").join(" ");

        QString copyright;
        if (!project->values("QMAKE_TARGET_COPYRIGHT").isEmpty())
            copyright = project->values("QMAKE_TARGET_COPYRIGHT").join(" ");

        QString productName;
        if (!project->values("QMAKE_TARGET_PRODUCT").isEmpty())
            productName = project->values("QMAKE_TARGET_PRODUCT").join(" ");
        else
            productName = project->values("TARGET").first();

        QString originalName = project->values("TARGET").first() +
                               project->values("TARGET_EXT").first();
        int rcLang =
            project->intValue("RC_LANG", 1033); // default: English(USA)
        int rcCodePage =
            project->intValue("RC_CODEPAGE", 1200); // default: Unicode
    }
}