# Mapping from Latvian method names to English equivalents
METHOD_MAPPING = {
    # Image loading methods
    'IelasitAtteluClick': 'LoadImageClick',
    'IelasitAttelu': 'LoadImage',
    
    # Date and coordinate setting methods
    'UzstaditDatumu': 'SetDate',
    'UzstaditKoordinates': 'SetCoordinates', 
    'UzstaditEpilinijuAugstumu': 'SetEpilineHeight',
    
    # Camera calibration methods
    'KalibretKameruClick': 'CalibrateCameraClick',
    'KamerasKalibracijasParametri': 'CameraCalibrationParameters',
    
    # Project management methods
    'NolasitProjektu': 'LoadProject',
    'NolasitProjektu2': 'LoadProject2',
    'SaglabatProjektu': 'SaveProject',
    
    # Camera file management
    'IelasitKameru': 'LoadCamera',
    'SaglabatKameru': 'SaveCamera',
    
    # Display methods
    'ZimetAltAzClick': 'DrawAltAzClick',
    'ZimetAtteluClick': 'DrawImageClick', 
    'ZimetAttelu': 'DrawImage',
    'ZimetKontrolpunktuAugstumus': 'DrawControlPointHeights',
    'ZimetAugstumuKarti': 'DrawHeightMap',
    
    # Star digitization methods
    'StartCiparotZvaigznes': 'StartDigitizeStars',
    'StopCiparotZvaigznes': 'StopDigitizeStars',
    'CiparotZvaigznesClick': 'DigitizeStarsClick',
    
    # Projection methods
    'ProjicetClick': 'ProjectClick',
    'ProjicetNoKartesClick': 'ProjectFromMapClick',
    'Projicet': 'Project',
    'ProjicetVidejotuAttelu': 'ProjectAveragedImage',
    'SaglabatProjicetoAttelu': 'SaveProjectedImage',
    
    # Region management
    'MainitApgabalu': 'ChangeRegion',
    'KartesApgabals': 'MapRegion',
    
    # Control point methods
    'CiparotKontrolpunktusClick': 'DigitizeControlPointsClick',
    'StartCiparotKontrolpunkti': 'StartDigitizeControlPoints',
    'StopCiparotKontrolpunkti': 'StopDigitizeControlPoints', 
    'IelasitKontrolpunktus': 'LoadControlPoints',
    
    # Height map methods
    'IzveidotAugstumuKarti': 'CreateHeightMap',
    'SaglabatAugstumuKarti': 'SaveHeightMap',
    'IelasitAugstumuKarti': 'LoadHeightMap',
    
    # Helper methods (keeping English names)
    'pelekot': 'update_ui_state',
    'distance': 'distance',
    'plotProjicet': 'plot_projection',
    'disconnect_meerit': 'disconnect_measurement',
    'connect_meerit': 'connect_measurement',
    'plotMatches': 'plot_matches',
    'onclick_ciparotzvaigznes': 'onclick_digitize_stars',
    'onclick_ciparotkontrolpunktus': 'onclick_digitize_control_points',
    'onclick_meritattalumu': 'onclick_measure_distance',
    'move_meritattalumu': 'move_measure_distance'
}