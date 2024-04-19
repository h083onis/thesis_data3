import socket
import sys
import eventlet
import eventlet . wsgi
import greenlet
from paste import deploy
import routes . middleware
import webob . dec
import webob . exc
from cinder import exception
from cinder import flags
from cinder . openstack . common import log as logging
from cinder import utils
FLAGS = flags . FLAGS
LOG = logging . getLogger ( __name__ )
class Server ( object ) :
    default_pool_size = 1000
    def __init__ ( self , name , app , host = None , port = None , pool_size = None , protocol = eventlet . wsgi . HttpProtocol ) :
        self . name = name
        self . app = app
        self . _host = host or "0.0.0.0"
        self . _port = port or 0
        self . _server = None
        self . _socket = None
        self . _protocol = protocol
        self . _pool = eventlet . GreenPool ( pool_size or self . default_pool_size )
        self . _logger = logging . getLogger ( "eventlet.wsgi.server" )
        self . _wsgi_logger = logging . WritableLogger ( self . _logger )
    def _start ( self ) :
        eventlet . wsgi . server ( self . _socket , self . app , protocol = self . _protocol , custom_pool = self . _pool , log = self . _wsgi_logger )
    def start ( self , backlog = 128 ) :
        if backlog < 1 :
            raise exception . InvalidInput ( reason = 'The backlog must be more than 1' )
        bind_addr = ( self . _host , self . _port )
        try :
            info = socket . getaddrinfo ( bind_addr [ 0 ] , bind_addr [ 1 ] , socket . AF_UNSPEC , socket . SOCK_STREAM ) [ 0 ]
            family = info [ 0 ]
            bind_addr = info [ - 1 ]
        except Exception :
            family = socket . AF_INET
        self . _socket = eventlet . listen ( bind_addr , family , backlog = backlog )
        self . _server = eventlet . spawn ( self . _start )
        ( self . _host , self . _port ) = self . _socket . getsockname ( ) [ 0 : 2 ]
        LOG . info ( _ ( "Started %(name)s on %(_host)s:%(_port)s" ) % self . __dict__ )
    @ property
    def host ( self ) :
        return self . _socket . getsockname ( ) [ 0 ] if self . _socket else self . _host
    @ property
    def port ( self ) :
        return self . _socket . getsockname ( ) [ 1 ] if self . _socket else self . _port
    def stop ( self ) :
        LOG . info ( _ ( "Stopping WSGI server." ) )
        self . _server . kill ( )
    def wait ( self ) :
        try :
            self . _server . wait ( )
        except greenlet . GreenletExit :
            LOG . info ( _ ( "WSGI server has stopped." ) )
class Request ( webob . Request ) :
    pass
class Application ( object ) :
    @ classmethod
    def factory ( cls , global_config , ** local_config ) :
        return cls ( ** local_config )
    def __call__ ( self , environ , start_response ) :
        raise NotImplementedError ( _ ( 'You must implement __call__' ) )
class Middleware ( Application ) :
    @ classmethod
    def factory ( cls , global_config , ** local_config ) :
        def _factory ( app ) :
            return cls ( app , ** local_config )
        return _factory
    def __init__ ( self , application ) :
        self . application = application
    def process_request ( self , req ) :
        return None
    def process_response ( self , response ) :
        return response
    @ webob . dec . wsgify ( RequestClass = Request )
    def __call__ ( self , req ) :
        response = self . process_request ( req )
        if response :
            return response
        response = req . get_response ( self . application )
        return self . process_response ( response )
class Debug ( Middleware ) :
    @ webob . dec . wsgify ( RequestClass = Request )
    def __call__ ( self , req ) :
        print ( '*' * 40 ) + ' REQUEST ENVIRON'
        for key , value in req . environ . items ( ) :
            print key , '=' , value
        print
        resp = req . get_response ( self . application )
        print ( '*' * 40 ) + ' RESPONSE HEADERS'
        for ( key , value ) in resp . headers . iteritems ( ) :
            print key , '=' , value
        print
        resp . app_iter = self . print_generator ( resp . app_iter )
        return resp
    @ staticmethod
    def print_generator ( app_iter ) :
        print ( '*' * 40 ) + ' BODY'
        for part in app_iter :
            sys . stdout . write ( part )
            sys . stdout . flush ( )
            yield part
        print
class Router ( object ) :
    def __init__ ( self , mapper ) :
        self . map = mapper
        self . _router = routes . middleware . RoutesMiddleware ( self . _dispatch , self . map )
    @ webob . dec . wsgify ( RequestClass = Request )
    def __call__ ( self , req ) :
        return self . _router
    @ staticmethod
    @ webob . dec . wsgify ( RequestClass = Request )
    def _dispatch ( req ) :
        match = req . environ [ 'wsgiorg.routing_args' ] [ 1 ]
        if not match :
            return webob . exc . HTTPNotFound ( )
        app = match [ 'controller' ]
        return app
class Loader ( object ) :
    def __init__ ( self , config_path = None ) :
        config_path = config_path or FLAGS . api_paste_config
        self . config_path = utils . find_config ( config_path )
    def load_app ( self , name ) :
        try :
            return deploy . loadapp ( "config:%s" % self . config_path , name = name )
        except LookupError as err :
            LOG . error ( err )
            raise exception . PasteAppNotFound ( name = name , path = self . config_path )
