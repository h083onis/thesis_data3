










































#include   " dbusconnection _ p . h "

#include   < QtDBus / QDBusMessage >
#include   < qdebug . h >

QT_BEGIN_NAMESPACE








DBusConnection :: DBusConnection ( )
     :   dbusConnection ( connectDBus ( ) )
{ }

QDBusConnection   DBusConnection :: connectDBus ( )
{
     QString   address   =   getAccessibilityBusAddress ( ) ;

     if   ( ! address . isEmpty ( ) )   {
         QDBusConnection   c   =   QDBusConnection :: connectToBus ( address ,   QStringLiteral ( " a11y " ) ) ;
         if   ( c . isConnected ( ) )   {
             qDebug ( )   <<   " Connected ▁ to ▁ accessibility ▁ bus ▁ at : ▁ "   <<   address ;
             return   c ;
         }
         qWarning ( " Found ▁ Accessibility ▁ DBus ▁ address ▁ but ▁ cannot ▁ connect . ▁ Falling ▁ back ▁ to ▁ session ▁ bus . " ) ;
     }   else   {
         qWarning ( " Accessibility ▁ DBus ▁ not ▁ found . ▁ Falling ▁ back ▁ to ▁ session ▁ bus . " ) ;
     }

     QDBusConnection   c   =   QDBusConnection :: sessionBus ( ) ;
     if   ( ! c . isConnected ( ) )   {
         qWarning ( " Could ▁ not ▁ connect ▁ to ▁ DBus . " ) ;
     }
     return   QDBusConnection :: sessionBus ( ) ;
}

QString   DBusConnection :: getAccessibilityBusAddress ( )   const
{
     QDBusConnection   c   =   QDBusConnection :: sessionBus ( ) ;

     QDBusMessage   m   =   QDBusMessage :: createMethodCall ( QLatin1String ( " org . a11y . Bus " ) ,
                                                     QLatin1String ( " / org / a11y / bus " ) ,
                                                     QLatin1String ( " org . a11y . Bus " ) ,   QLatin1String ( " GetAddress " ) ) ;
     QDBusMessage   reply   =   c . call ( m ) ;
     if   ( reply . type ( )   ==   QDBusMessage :: ErrorMessage )   {
         qWarning ( )   <<   " Qt ▁ at - spi : ▁ error ▁ getting ▁ the ▁ accessibility ▁ dbus ▁ address : ▁ "   <<   reply . errorMessage ( ) ;
         return   QString ( ) ;
     }

     QString   busAddress   =   reply . arguments ( ) . at ( 0 ) . toString ( ) ;
     qDebug ( )   <<   " Got ▁ bus ▁ address : ▁ "   <<   busAddress ;
     return   busAddress ;
}




QDBusConnection   DBusConnection :: connection ( )   const
{
     return   dbusConnection ;
}

QT_END_NAMESPACE