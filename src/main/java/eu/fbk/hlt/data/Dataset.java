package eu.fbk.hlt.data;

import java.io.*;
import java.net.URISyntaxException;
import java.util.zip.GZIPInputStream;

/**
 * The main entry point for downloading and loading into memory the datasets
 *
 * @author Yaroslav Nechaev (remper@me.com)
 */
public abstract class Dataset {
    protected DatasetMetaInfo info;
    protected File source;

    public Dataset(DatasetMetaInfo info) throws URISyntaxException {
        assert info.isOffline;
        this.info = info;
        this.source = new File(this.info.location.toURI());
        parse();
    }

    public abstract void parse();

    protected LineNumberReader getReader() throws IOException {
        InputStream stream = new FileInputStream(source);
        switch (this.info.compression) {
            case GZ:
                stream = new GZIPInputStream(stream);
                break;
            case PLAIN:
                //No need to modify stream here
                break;
        }
        return new LineNumberReader(new InputStreamReader(stream));
    }
}
