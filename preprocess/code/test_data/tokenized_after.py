import   errno
import   os
import   socket
import   ssl
import   sys
import   time
import   eventlet
import   eventlet . wsgi
import   greenlet
from   paste   import   deploy
import   routes . middleware
import   webob . dec
import   webob . exc
from   cinder   import   exception
from   cinder   import   flags
from   cinder . openstack . common   import   cfg
from   cinder . openstack . common   import   log   as   logging
from   cinder   import   utils
socket_opts   =   [
     cfg . IntOpt ( ' backlog ' ,
                default = 4096 ,
                help = " Number ▁ of ▁ backlog ▁ requests ▁ to ▁ configure ▁ the ▁ socket ▁ with " ) ,
     cfg . IntOpt ( ' tcp _ keepidle ' ,
                default = 600 ,
                help = " Sets ▁ the ▁ value ▁ of ▁ TCP _ KEEPIDLE ▁ in ▁ seconds ▁ for ▁ each ▁ "
                     " server ▁ socket . ▁ Not ▁ supported ▁ on ▁ OS ▁ X . " ) ,
     cfg . StrOpt ( ' ssl _ ca _ file ' ,
                default = None ,
                help = " CA ▁ certificate ▁ file ▁ to ▁ use ▁ to ▁ verify ▁ "
                     " connecting ▁ clients " ) ,
     cfg . StrOpt ( ' ssl _ cert _ file ' ,
                default = None ,
                help = " Certificate ▁ file ▁ to ▁ use ▁ when ▁ starting ▁ "
                     " the ▁ server ▁ securely " ) ,
     cfg . StrOpt ( ' ssl _ key _ file ' ,
                default = None ,
                help = " Private ▁ key ▁ file ▁ to ▁ use ▁ when ▁ starting ▁ "
                     " the ▁ server ▁ securely " ) ,
]
CONF   =   cfg . CONF
CONF . register_opts ( socket_opts )
FLAGS   =   flags . FLAGS
LOG   =   logging . getLogger ( __name__ )
class   Server ( object ) :
     default_pool_size   =   1000
     def   __init__ ( self ,   name ,   app ,   host = None ,   port = None ,   pool_size = None ,
                  protocol = eventlet . wsgi . HttpProtocol ) :
         self . name   =   name
         self . app   =   app
         self . _host   =   host   or   "0.0.0.0"
         self . _port   =   port   or   0
         self . _server   =   None
         self . _socket   =   None
         self . _protocol   =   protocol
         self . _pool   =   eventlet . GreenPool ( pool_size   or   self . default_pool_size )
         self . _logger   =   logging . getLogger ( " eventlet . wsgi . server " )
         self . _wsgi_logger   =   logging . WritableLogger ( self . _logger )
     def   _get_socket ( self ,   host ,   port ,   backlog ) :
         bind_addr   =   ( host ,   port )
         try :
             info   =   socket . getaddrinfo ( bind_addr [ 0 ] ,
                                       bind_addr [ 1 ] ,
                                       socket . AF_UNSPEC ,
                                       socket . SOCK_STREAM ) [ 0 ]
             family   =   info [ 0 ]
             bind_addr   =   info [ - 1 ]
         except   Exception :
             family   =   socket . AF_INET
         cert_file   =   CONF . ssl_cert_file
         key_file   =   CONF . ssl_key_file
         ca_file   =   CONF . ssl_ca_file
         use_ssl   =   cert_file   or   key_file
         if   cert_file   and   not   os . path . exists ( cert_file ) :
             raise   RuntimeError ( _ ( " Unable ▁ to ▁ find ▁ cert _ file ▁ : ▁ % s " )   %   cert_file )
         if   ca_file   and   not   os . path . exists ( ca_file ) :
             raise   RuntimeError ( _ ( " Unable ▁ to ▁ find ▁ ca _ file ▁ : ▁ % s " )   %   ca_file )
         if   key_file   and   not   os . path . exists ( key_file ) :
             raise   RuntimeError ( _ ( " Unable ▁ to ▁ find ▁ key _ file ▁ : ▁ % s " )   %   key_file )
         if   use_ssl   and   ( not   cert_file   or   not   key_file ) :
             raise   RuntimeError ( _ ( " When ▁ running ▁ server ▁ in ▁ SSL ▁ mode , ▁ you ▁ must ▁ "
                                  " specify ▁ both ▁ a ▁ cert _ file ▁ and ▁ key _ file ▁ "
                                  " option ▁ value ▁ in ▁ your ▁ configuration ▁ file " ) )
         def   wrap_ssl ( sock ) :
             ssl_kwargs   =   {
                 ' server _ side ' :   True ,
                 ' certfile ' :   cert_file ,
                 ' keyfile ' :   key_file ,
                 ' cert _ reqs ' :   ssl . CERT_NONE ,
             }
             if   CONF . ssl_ca_file :
                 ssl_kwargs [ ' ca _ certs ' ]   =   ca_file
                 ssl_kwargs [ ' cert _ reqs ' ]   =   ssl . CERT_REQUIRED
             return   ssl . wrap_socket ( sock ,   ** ssl_kwargs )
         sock   =   None
         retry_until   =   time . time ( )   +   30
         while   not   sock   and   time . time ( )   <   retry_until :
             try :
                 sock   =   eventlet . listen ( bind_addr ,
                                        backlog = backlog ,
                                        family = family )
                 if   use_ssl :
                     sock   =   wrap_ssl ( sock )
             except   socket . error ,   err :
                 if   err . args [ 0 ]   !=   errno . EADDRINUSE :
                     raise
                 eventlet . sleep ( 0.1 )
         if   not   sock :
             raise   RuntimeError ( _ ( " Could ▁ not ▁ bind ▁ to ▁ % ( host ) s : % ( port ) s ▁ "
                                " after ▁ trying ▁ for ▁ 30 ▁ seconds " )   %
                                { ' host ' :   host ,   ' port ' :   port } )
         sock . setsockopt ( socket . SOL_SOCKET ,   socket . SO_REUSEADDR ,   1 )
         sock . setsockopt ( socket . SOL_SOCKET ,   socket . SO_KEEPALIVE ,   1 )
         if   hasattr ( socket ,   ' TCP _ KEEPIDLE ' ) :
             sock . setsockopt ( socket . IPPROTO_TCP ,
                             socket . TCP_KEEPIDLE ,
                             CONF . tcp_keepidle )
         return   sock
     def   _start ( self ) :
         eventlet . wsgi . server ( self . _socket ,
                              self . app ,
                              protocol = self . _protocol ,
                              custom_pool = self . _pool ,
                              log = self . _wsgi_logger )
     def   start ( self ,   backlog = 128 ) :
         if   backlog   <   1 :
             raise   exception . InvalidInput (
                 reason = ' The ▁ backlog ▁ must ▁ be ▁ more ▁ than ▁ 1' )
         self . _socket   =   self . _get_socket ( self . _host ,
                                         self . _port ,
                                         backlog = backlog )
         self . _server   =   eventlet . spawn ( self . _start )
         ( self . _host ,   self . _port )   =   self . _socket . getsockname ( ) [ 0 : 2 ]
         LOG . info ( _ ( " Started ▁ % ( name ) s ▁ on ▁ % ( _ host ) s : % ( _ port ) s " )   %   self . __dict__ )
     @ property
     def   host ( self ) :
         return   self . _host
     @ property
     def   port ( self ) :
         return   self . _port
     def   stop ( self ) :
         LOG . info ( _ ( " Stopping ▁ WSGI ▁ server . " ) )
         self . _server . kill ( )
     def   wait ( self ) :
         try :
             self . _server . wait ( )
         except   greenlet . GreenletExit :
             LOG . info ( _ ( " WSGI ▁ server ▁ has ▁ stopped . " ) )
class   Request ( webob . Request ) :
     pass
class   Application ( object ) :
     @ classmethod
     def   factory ( cls ,   global_config ,   ** local_config ) :
         return   cls ( ** local_config )
     def   __call__ ( self ,   environ ,   start_response ) :
         raise   NotImplementedError ( _ ( ' You ▁ must ▁ implement ▁ _ _ call _ _ ' ) )
class   Middleware ( Application ) :
     @ classmethod
     def   factory ( cls ,   global_config ,   ** local_config ) :
         def   _factory ( app ) :
             return   cls ( app ,   ** local_config )
         return   _factory
     def   __init__ ( self ,   application ) :
         self . application   =   application
     def   process_request ( self ,   req ) :
         return   None
     def   process_response ( self ,   response ) :
         return   response
     @ webob . dec . wsgify ( RequestClass = Request )
     def   __call__ ( self ,   req ) :
         response   =   self . process_request ( req )
         if   response :
             return   response
         response   =   req . get_response ( self . application )
         return   self . process_response ( response )
class   Debug ( Middleware ) :
     @ webob . dec . wsgify ( RequestClass = Request )
     def   __call__ ( self ,   req ) :
         print   ( ' * '   *   40 )   +   ' ▁ REQUEST ▁ ENVIRON '
         for   key ,   value   in   req . environ . items ( ) :
             print   key ,   ' = ' ,   value
         print
         resp   =   req . get_response ( self . application )
         print   ( ' * '   *   40 )   +   ' ▁ RESPONSE ▁ HEADERS '
         for   ( key ,   value )   in   resp . headers . iteritems ( ) :
             print   key ,   ' = ' ,   value
         print
         resp . app_iter   =   self . print_generator ( resp . app_iter )
         return   resp
     @ staticmethod
     def   print_generator ( app_iter ) :
         print   ( ' * '   *   40 )   +   ' ▁ BODY '
         for   part   in   app_iter :
             sys . stdout . write ( part )
             sys . stdout . flush ( )
             yield   part
         print
class   Router ( object ) :
     def   __init__ ( self ,   mapper ) :
         self . map   =   mapper
         self . _router   =   routes . middleware . RoutesMiddleware ( self . _dispatch ,
                                                           self . map )
     @ webob . dec . wsgify ( RequestClass = Request )
     def   __call__ ( self ,   req ) :
         return   self . _router
     @ staticmethod
     @ webob . dec . wsgify ( RequestClass = Request )
     def   _dispatch ( req ) :
         match   =   req . environ [ ' wsgiorg . routing _ args ' ] [ 1 ]
         if   not   match :
             return   webob . exc . HTTPNotFound ( )
         app   =   match [ ' controller ' ]
         return   app
class   Loader ( object ) :
     def   __init__ ( self ,   config_path = None ) :
         config_path   =   config_path   or   FLAGS . api_paste_config
         self . config_path   =   utils . find_config ( config_path )
     def   load_app ( self ,   name ) :
         try :
             return   deploy . loadapp ( " config : % s "   %   self . config_path ,   name = name )
         except   LookupError   as   err :
             LOG . error ( err )