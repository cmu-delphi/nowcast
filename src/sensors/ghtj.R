## Synopsis: Make single national or HHS level prediction; this is to be called
## by a justinfun() from within get_ghtj() in update_sensor.py;

library(ghtModel)

## Global settings:
args = commandArgs(trailingOnly=TRUE)
start.date = "201001" ## Fix and don't change
end.date = args[2] ##"201744"
curr.date = end.date  #"201744"
train.duration = 52*2
how.many.weeks.back = 52*2
fluview.area = args[1] ##"nat"
datadir = outputdir = "/home/justin/repos/ghtModel/output" ## "/home/shyun/output/"
                                                           ## ## Need to set an
                                                           ## absolute path

states.list = list(hhs1 = c('CT','ME','MA','NH','RI','VT'), hhs2 = c('NJ','NY'),
                   hhs3 = c('DE','DC','MD','PA','VA','WV'),
                   hhs4 = c('AL','FL','GA','KY','MS','NC','SC','TN'),
                   hhs5 = c('IL','IN','MI','MN','OH','WI'),
                   hhs6 = c('AR','LA','NM','OK','TX'),
                   hhs7 = c('IA','KS','MO','NE'),
                   hhs8 = c('CO','MT','ND','SD','UT','WY'),
                   hhs9 = c('AZ','CA','HI','NV'), hhs10= c('AK','ID','OR','WA'))

## |predfilename| is file containing the single prediction.
## |topfiftyfilename| is the file containing the fifty top terms to be used for prediction
## |destfilename| is the intermediate file that contains the GHT search query volumes
topfiftyfilename = "topfifty.txt"
predfilename = paste0("ghtpred-", fluview.area, "-", curr.date, ".txt")


## If regional predictions are being made, use this:
if(fluview.area %in% names(states.list)){
    states = states.list[[fluview.area]]
    my.ght.table.list = list()
    for(ii in 1:length(states)){
        ## Get ght table (via python script and external saving of a file)
        state = states[ii]
        destfilename = paste0("ght-", state, "-", curr.date, ".csv")
        query_and_save_ght(mycommand = "python",
                           py.script = "/home/justin/repos/ghtModel/main/query-ght/get_ght.py",
                           ## py.script = "../python/get_ght.py",
                           destfilename = destfilename,
                           termfilename =  topfiftyfilename,
                           start.date = start.date, end.date = end.date,
                           datadir = datadir,
                           fluview.area = state)
        my.ght.table.list[[ii]] = get_ght_from_file(state, dir=datadir, file=destfilename)
    }
    names(my.ght.table.list) = states
    my.ght.table = Reduce(function(...) merge(..., all = TRUE, by = "epiweek"), ## incomplete..
                          my.ght.table.list)
} else if (fluview.area == "nat"){
    destfilename = paste0("ght-", fluview.area, "-", curr.date, ".csv")
    query_and_save_ght(mycommand = "python",
                       py.script = "/home/justin/repos/ghtModel/main/query-ght/get_ght.py",
                       ## py.script = "../python/get_ght.py",
                       destfilename = destfilename,
                       termfilename =  topfiftyfilename,
                       start.date = start.date, end.date = end.date,
                       datadir = datadir,
                       fluview.area = fluview.area)
    my.ght.table = get_ght_from_file(fluview.area, dir=datadir, file=destfilename)
}

## Make latest prediction and write to a file.
all.epiweeks = make_epiweeks(as.numeric(start.date), as.numeric(end.date))
pred.df =  makepred(length(all.epiweeks), all.epiweeks, train.duration, fluview.area, my.ght.table)
write(pred.df[,"pred"], file=file.path(outputdir, predfilename))
