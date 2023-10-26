<?php
session_start();
$db = new PDO('sqlite:bazakina.db');


if(isset($_GET['id_seans']))
  {
    $id_seans=intval($_GET['id_seans']);
   if($id_seans>0) 
    { 
        $sql="DELETE FROM seanse WHERE id_seans=$id_seans"; 
        $db->exec($sql); 
    
        header("location: dodajseans.php");
    }
    else 
    {
      print "Wybrany seans nie istnieje w bazie.";
        header("location: dodajseans.php");
    }
  }

?>