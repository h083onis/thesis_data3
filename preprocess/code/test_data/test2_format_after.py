import math
import os
import re
from cinder import exception
from cinder import flags
from cinder.image import image_utils
from cinder.openstack.common import cfg
from cinder.openstack.common import log as logging
from cinder import utils
from cinder.volume import driver
from cinder.volume import iscsi
LOG = logging.getLogger ( __name__ )
volume_opts = [ cfg.StrOpt ( "volume_group" , default = "cinder-volumes" , help = "Name for the VG that will contain exported volumes" , ) , cfg.StrOpt ( "volume_clear" , default = "zero" , help = "Method used to wipe old volumes (valid options are: " "none, zero, shred)" , ) , cfg.IntOpt ( "volume_clear_size" , default = 0 , help = "Size in MiB to wipe at start of old volumes.0 => all" , ) , cfg.IntOpt ( "lvm_mirrors" , default = 0 , help = "If set, create lvms with multiple mirrors.Note that " "this requires lvm_mirrors + 2 pvs with available space" , ) , ]
FLAGS = flags.FLAGS
FLAGS.register_opts ( volume_opts )
class LVMVolumeDriver ( driver.VolumeDriver ) :
    def __init__ ( self , * args , ** kwargs ) :
        super ( LVMVolumeDriver , self ).__init__ ( * args , ** kwargs )
    def check_for_setup_error ( self ) :
        out , err = self._execute ( "vgs" , "--noheadings" , "-o" , "name" , run_as_root = True )
        volume_groups = out.split ( )
        if not FLAGS.volume_group in volume_groups :
            exception_message = _ ( "volume group %s doesn't exist" ) % FLAGS.volume_group
            raise exception.VolumeBackendAPIException ( data = exception_message )
    def _create_volume ( self , volume_name , sizestr ) :
        cmd = [ "lvcreate" , "-L" , sizestr , "-n" , volume_name , FLAGS.volume_group ]
        if FLAGS.lvm_mirrors :
            cmd += [ "-m" , FLAGS.lvm_mirrors , "--nosync" ]
            terras = int ( sizestr [ : - 1 ] ) / 1024.0
            if terras >= 1.5 :
                rsize = int ( 2 ** math.ceil ( math.log ( terras ) / math.log ( 2 ) ) )
                cmd += [ "-R" , str ( rsize ) ]
        self._try_execute ( * cmd , run_as_root = True )
    def _copy_volume ( self , srcstr , deststr , size_in_g , clearing = False ) :
        extra_flags = [ "iflag=direct" , "oflag=direct" ]
        try :
            self._execute ( "dd" , "count=0" , "if=%s" % srcstr , "of=%s" % deststr , * extra_flags , run_as_root = True )
        except exception.ProcessExecutionError :
            extra_flags = [ ]
        if clearing and not extra_flags :
            extra_flags.append ( "conv=fdatasync" )
        self._execute ( "dd" , "if=%s" % srcstr , "of=%s" % deststr , "count=%d" % ( size_in_g * 1024 ) , "bs=1M" , * extra_flags , run_as_root = True )
    def _volume_not_present ( self , volume_name ) :
        path_name = "%s/%s" % ( FLAGS.volume_group , volume_name )
        try :
            self._try_execute ( "lvdisplay" , path_name , run_as_root = True )
        except Exception as e :
            return True
        return False
    def _delete_volume ( self , volume , size_in_g ) :
        dev_path = self.local_path ( volume )
        if os.path.exists ( dev_path ) :
            self.clear_volume ( volume )
        self._try_execute ( "lvremove" , "-f" , "%s/%s" % ( FLAGS.volume_group , self._escape_snapshot ( volume [ "name" ] ) ) , run_as_root = True , )
    def _sizestr ( self , size_in_g ) :
        if int ( size_in_g ) == 0 :
            return "100M"
        return "%sG" % size_in_g
    def _escape_snapshot ( self , snapshot_name ) :
        if not snapshot_name.startswith ( "snapshot" ) :
            return snapshot_name
        return "_" + snapshot_name
    def create_volume ( self , volume ) :
        self._create_volume ( volume [ "name" ] , self._sizestr ( volume [ "size" ] ) )
    def create_volume_from_snapshot ( self , volume , snapshot ) :
        self._create_volume ( volume [ "name" ] , self._sizestr ( volume [ "size" ] ) )
        self._copy_volume ( self.local_path ( snapshot ) , self.local_path ( volume ) , snapshot [ "volume_size" ] )
    def delete_volume ( self , volume ) :
        if self._volume_not_present ( volume [ "name" ] ) :
            return True
        out , err = self._execute ( "lvdisplay" , "--noheading" , "-C" , "-o" , "Attr" , "%s/%s" % ( FLAGS.volume_group , volume [ "name" ] ) , run_as_root = True , )
        if out :
            out = out.strip ( )
            if ( out [ 0 ] == "o" ) or ( out [ 0 ] == "O" ) :
                raise exception.VolumeIsBusy ( volume_name = volume [ "name" ] )
        self._delete_volume ( volume , volume [ "size" ] )
    def clear_volume ( self , volume ) :
        vol_path = self.local_path ( volume )
        size_in_g = volume.get ( "size" )
        size_in_m = FLAGS.volume_clear_size
        if not size_in_g :
            return
        if FLAGS.volume_clear == "none" :
            return
        LOG.info ( _ ( "Performing secure delete on volume: %s" ) % volume [ "id" ] )
        if FLAGS.volume_clear == "zero" :
            if size_in_m == 0 :
                return self._copy_volume ( "/dev/zero" , vol_path , size_in_g , clearing = True )
            else :
                clear_cmd = [ "shred" , "-n0" , "-z" , "-s%dMiB" % size_in_m ]
        elif FLAGS.volume_clear == "shred" :
            clear_cmd = [ "shred" , "-n3" ]
            if size_in_m :
                clear_cmd.append ( "-s%dMiB" % size_in_m )
        else :
            LOG.error ( _ ( "Error unrecognized volume_clear option: %s" ) , FLAGS.volume_clear )
            return
        clear_cmd.append ( vol_path )
        self._execute ( * clear_cmd , run_as_root = True )
    def create_snapshot ( self , snapshot ) :
        orig_lv_name = "%s/%s" % ( FLAGS.volume_group , snapshot [ "volume_name" ] )
        self._try_execute ( "lvcreate" , "-L" , self._sizestr ( snapshot [ "volume_size" ] ) , "--name" , self._escape_snapshot ( snapshot [ "name" ] ) , "--snapshot" , orig_lv_name , run_as_root = True , )
    def delete_snapshot ( self , snapshot ) :
        if self._volume_not_present ( self._escape_snapshot ( snapshot [ "name" ] ) ) :
            return True
        self._delete_volume ( snapshot , snapshot [ "volume_size" ] )
    def local_path ( self , volume ) :
        escaped_group = FLAGS.volume_group.replace ( "-" , "--" )
        escaped_name = self._escape_snapshot ( volume [ "name" ] ).replace ( "-" , "--" )
        return "/dev/mapper/%s-%s" % ( escaped_group , escaped_name )
    def copy_image_to_volume ( self , context , volume , image_service , image_id ) :
        image_utils.fetch_to_raw ( context , image_service , image_id , self.local_path ( volume ) )
    def copy_volume_to_image ( self , context , volume , image_service , image_id ) :
        volume_path = self.local_path ( volume )
        with utils.temporary_chown ( volume_path ) :
            with utils.file_open ( volume_path ) as volume_file :
                image_service.update ( context , image_id , { } , volume_file )
    def clone_image ( self , volume , image_location ) :
        return False
class LVMISCSIDriver ( LVMVolumeDriver , driver.ISCSIDriver ) :
    def __init__ ( self , * args , ** kwargs ) :
        self.tgtadm = iscsi.get_target_admin ( )
        super ( LVMISCSIDriver , self ).__init__ ( * args , ** kwargs )
    def set_execute ( self , execute ) :
        super ( LVMISCSIDriver , self ).set_execute ( execute )
        self.tgtadm.set_execute ( execute )
    def ensure_export ( self , context , volume ) :
        if not isinstance ( self.tgtadm , iscsi.TgtAdm ) :
            try :
                iscsi_target = self.db.volume_get_iscsi_target_num ( context , volume [ "id" ] )
            except exception.NotFound :
                LOG.info ( _ ( "Skipping ensure_export.No iscsi_target " "provisioned for volume: %s" ) , volume [ "id" ] , )
                return
        else :
            iscsi_target = 1
        old_name = None
        volume_name = volume [ "name" ]
        if ( volume [ "provider_location" ] is not None and volume [ "name" ] not in volume [ "provider_location" ] ) :
            msg = _ ( "Detected inconsistency in provider_location id" )
            LOG.debug ( msg )
            old_name = self._fix_id_migration ( context , volume )
            if "in-use" in volume [ "status" ] :
                volume_name = old_name
                old_name = None
        iscsi_name = "%s%s" % ( FLAGS.iscsi_target_prefix , volume_name )
        volume_path = "/dev/%s/%s" % ( FLAGS.volume_group , volume_name )
        self.tgtadm.create_iscsi_target ( iscsi_name , iscsi_target , 0 , volume_path , check_exit_code = False , old_name = old_name , )
    def _fix_id_migration ( self , context , volume ) :
        model_update = { }
        pattern = re.compile ( r":|\s" )
        fields = pattern.split ( volume [ "provider_location" ] )
        old_name = fields [ 3 ]
        volume [ "provider_location" ] = volume [ "provider_location" ].replace ( old_name , volume [ "name" ] )
        model_update [ "provider_location" ] = volume [ "provider_location" ]
        self.db.volume_update ( context , volume [ "id" ] , model_update )
        start = os.getcwd ( )
        os.chdir ( "/dev/%s" % FLAGS.volume_group )
        try :
            ( out , err ) = self._execute ( "readlink" , old_name )
        except exception.ProcessExecutionError :
            link_path = "/dev/%s/%s" % ( FLAGS.volume_group , old_name )
            LOG.debug ( _ ( "Symbolic link %s not found" ) % link_path )
            os.chdir ( start )
            return
        rel_path = out.rstrip ( )
        self._execute ( "ln" , "-s" , rel_path , volume [ "name" ] , run_as_root = True )
        os.chdir ( start )
        return old_name
    def _ensure_iscsi_targets ( self , context , host ) :
        if not isinstance ( self.tgtadm , iscsi.TgtAdm ) :
            host_iscsi_targets = self.db.iscsi_target_count_by_host ( context , host )
            if host_iscsi_targets >= FLAGS.iscsi_num_targets :
                return
            for target_num in xrange ( 1 , FLAGS.iscsi_num_targets + 1 ) :
                target = { "host" : host , "target_num" : target_num }
                self.db.iscsi_target_create_safe ( context , target )
    def create_export ( self , context , volume ) :
        iscsi_name = "%s%s" % ( FLAGS.iscsi_target_prefix , volume [ "name" ] )
        volume_path = "/dev/%s/%s" % ( FLAGS.volume_group , volume [ "name" ] )
        model_update = { }
        if not isinstance ( self.tgtadm , iscsi.TgtAdm ) :
            lun = 0
            self._ensure_iscsi_targets ( context , volume [ "host" ] )
            iscsi_target = self.db.volume_allocate_iscsi_target ( context , volume [ "id" ] , volume [ "host" ] )
        else :
            lun = 1
            iscsi_target = 0
        chap_username = utils.generate_username ( )
        chap_password = utils.generate_password ( )
        chap_auth = self._iscsi_authentication ( "IncomingUser" , chap_username , chap_password )
        tid = self.tgtadm.create_iscsi_target ( iscsi_name , iscsi_target , 0 , volume_path , chap_auth )
        model_update [ "provider_location" ] = self._iscsi_location ( FLAGS.iscsi_ip_address , tid , iscsi_name , lun )
        model_update [ "provider_auth" ] = self._iscsi_authentication ( "CHAP" , chap_username , chap_password )
        return model_update
    def remove_export ( self , context , volume ) :
        if not isinstance ( self.tgtadm , iscsi.TgtAdm ) :
            try :
                iscsi_target = self.db.volume_get_iscsi_target_num ( context , volume [ "id" ] )
            except exception.NotFound :
                LOG.info ( _ ( "Skipping remove_export.No iscsi_target " "provisioned for volume: %s" ) , volume [ "id" ] , )
                return
        else :
            iscsi_target = 0
        try :
            location = volume [ "provider_location" ].split ( " " )
            iqn = location [ 1 ]
            self.tgtadm.show_target ( iscsi_target , iqn = iqn )
        except Exception as e :
            LOG.info ( _ ( "Skipping remove_export.No iscsi_target " "is presently exported for volume: %s" ) , volume [ "id" ] , )
            return
        self.tgtadm.remove_iscsi_target ( iscsi_target , 0 , volume [ "id" ] )
    def get_volume_stats ( self , refresh = False ) :
        if refresh :
            self._update_volume_status ( )
        return self._stats
    def _update_volume_status ( self ) :
        LOG.debug ( _ ( "Updating volume status" ) )
        data = { }
        data [ "volume_backend_name" ] = "LVM_iSCSI"
        data [ "vendor_name" ] = "Open Source"
        data [ "driver_version" ] = "1.0"
        data [ "storage_protocol" ] = "iSCSI"
        data [ "total_capacity_gb" ] = 0
        data [ "free_capacity_gb" ] = 0
        data [ "reserved_percentage" ] = FLAGS.reserved_percentage
        data [ "QoS_support" ] = False
        try :
            out , err = self._execute ( "vgs" , "--noheadings" , "--nosuffix" , "--unit=G" , "-o" , "name,size,free" , FLAGS.volume_group , run_as_root = True , )
        except exception.ProcessExecutionError as exc :
            LOG.error ( _ ( "Error retrieving volume status: " ) , exc.stderr )
            out = False
        if out :
            volume = out.split ( )
            data [ "total_capacity_gb" ] = float ( volume [ 1 ] )
            data [ "free_capacity_gb" ] = float ( volume [ 2 ] )
        self._stats = data
    def _iscsi_location ( self , ip , target , iqn , lun = None ) :
        return "%s:%s,%s %s %s" % ( ip , FLAGS.iscsi_port , target , iqn , lun )
    def _iscsi_authentication ( self , chap , name , password ) :
        return "%s %s %s" % ( chap , name , password )
