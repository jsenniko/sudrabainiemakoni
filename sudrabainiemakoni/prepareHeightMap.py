import os, glob
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from sudrabainiemakoni.cloudimage import CloudImage, CloudImagePair
from sudrabainiemakoni.cloudimage import WebMercatorImage
from sudrabainiemakoni import plots, argumentsSM

def doProcessing(args):
    # izveidojam divus CloudImage tipa objektus, katru savam attēlam
    # Ielasam no faila, pirms tam referencētus attēlus
    cldim1 = CloudImage.load(args.loadProject1)
    cldim2 = CloudImage.load(args.loadProject2)
    # izveidojam CloudImagePair objektu, kas darbojas ar abiem attēliem
    imagepair = CloudImagePair(cldim1, cldim2)
    imagepair.LoadCorrespondances(args.loadControlPoints)

    # Atrodam kopīgo punktu koordinātes (platums, garums, augstums) (llh), attālumu starp stariem uz kopīgajiem punktiem (metros) (rayminimaldistance), (z_intrinsic_error) - potenciālā kļūda km/px (jo mazāka jo labāk - kopīgie punkti atrodas uz garāka epilīniju nogriežņa), valid - iezīme, vai punkts derīgs atbilstoši kritētijiem
    llh, rayminimaldistance, z_intrinsic_error, valid = imagepair.GetHeightPoints(*imagepair.correspondances)
    # sagatavojam punktu datus tabulas formā
    df_points = pd.DataFrame({'lat':llh[0], 'lon':llh[1], 'z':llh[2], 'Rdist':rayminimaldistance, 'zerr':z_intrinsic_error, 'valid':valid})
    if args.saveControlPoints is not None:
        df_points.to_csv(args.saveControlPoints, sep='\t')
    if args.plotControlPoints is not None:
        # izveidojam punktiem atbilstošos epilīniju nogriežņus augstumu intervālam 75 - 90 km
        z1, z2 = 75, 90
        # epilīnijas pirmajā attēlā, kas atbilst punktiem otrajā attēlā
        epilines = imagepair.GetEpilinesAtHeightInterval([z1,z2],imagepair.correspondances[1], False)
        # izvadām punktu attēlus
        fn=os.path.splitext(args.plotControlPoints)[0]
        plots.PlotValidHeightPoints(cldim1.imagearray,epilines,imagepair.correspondances[0] , llh[2], valid,
                                    filename = f"{fn}_1_2.jpg")
        plots.PlotValidHeightPoints(cldim1.imagearray,epilines,imagepair.correspondances[0] , llh[2], None,
                                    filename = f"{fn}_1_2_all.jpg")
        # epilīnijas otrajā attēlā, kas atbilst punktiem pirmajā attēlā
        epilines = imagepair.GetEpilinesAtHeightInterval([z1,z2],imagepair.correspondances[0], True)
        # izvadām punktu attēlus
        plots.PlotValidHeightPoints(cldim2.imagearray,epilines,imagepair.correspondances[1] , llh[2], valid,
                                    filename = f"{fn}_2_1.jpg")
        plots.PlotValidHeightPoints(cldim2.imagearray,epilines,imagepair.correspondances[1] , llh[2], None,
                                    filename = f"{fn}_2_1_all.jpg")
    # projicēšanas apgabals un izšķirtspēja
    lonmin, lonmax, latmin, latmax, horizontal_resolution_km = np.array(args.webMercParameters.split(',')).astype('float')
    # sagatavojam projicēšanas objektu
    webmerc = WebMercatorImage(cldim1, lonmin, lonmax, latmin, latmax, horizontal_resolution_km)

    # veidojam interpolēto augstumu sadalījumu ar *kriging* palīdzību, no derīgajiem punktiem
    heightgrid = webmerc.PrepareHeightMap(llh[1][valid],llh[0][valid],llh[2][valid])
    if args.saveHeightMap is not None:
        np.save(args.saveHeightMap, heightgrid)
    if args.saveGeoMap is not None:
        # augstumu sadalījums uz kartes vizuālai kontrolei
        map_lonmin, map_lonmax, map_latmin, map_latmax = np.array(args.mapBounds.split(',')).astype('float')
        pp=[[cldim1.location.lon.value, cldim1.location.lat.value]]
        #pp=np.array(list(zip(llh[1][valid],llh[0][valid])))
        fig, ax = plt.subplots(figsize=(16,9), facecolor='#FAFAFA')
        csl=plots.PlotReferencedImages(webmerc, [heightgrid],  camera_points=pp,
                                       outputFileName=None,
                                       lonmin=map_lonmin, lonmax=map_lonmax, latmin=map_latmin, latmax=map_latmax,
                                       showplot=True,
                                       alpha=0.8, ax=ax)
        import tilemapbase
        xy=np.array([tilemapbase.project(lon,lat) for lon,lat in zip(llh[1][valid],llh[0][valid])])
        cs=ax.scatter(xy[:,0],xy[:,1], c=llh[2][valid], norm=csl[0].norm, cmap=csl[0].cmap)
        fig.colorbar(csl[0])
        fig.savefig(args.saveGeoMap, bbox_inches='tight', dpi=300)
        plt.close()



if __name__ == "__main__":
    args = argumentsSM.parse_arguments_heightmap()
    print(args)
    doProcessing(args)